import re
import gc
import time
import copy
import torch
import logging
import importlib
import numpy as np
from z3 import *
from core.config import *
from multiprocessing import Pool
from utils.find import find_agent
from utils.sample import split_into_subsets
from planners.local.basic import LocalPlanner

logger = logging.getLogger(__name__)

# used for grounding
class PesudoAgent:
    def __init__(self, type, layer_id, concepts):
        self.type = type
        self.layer_id = layer_id
        self.type = concepts["type"]
        self.concepts = concepts

class Z3PlannerLocal(LocalPlanner):
    def __init__(self, yaml_path):        
        super().__init__(yaml_path)

    def _create_entities(self):
        # Create Z3 sorts for each entity type
        self.entity_types = []
        for entity_type in self.data["EntityTypes"]:
            # Create a Z3 sort (type) for each entity
            self.entity_types.append(entity_type)
        # Print the entity types
        entity_types_info = "\n".join(["- {}".format(entity) for entity in self.entity_types])
        logger.info("Number of Entity Types: {}\nEntity Types:\n{}".format(len(self.entity_types), entity_types_info))

    def _create_predicates(self):
        self.predicates = {}
        for pred_dict in self.data["Predicates"]:
            (pred_name, info), = pred_dict.items()
            method_name = info["method"].split('(')[0]
            arity = info["arity"]
            z3_func = None
            if arity == 1:
                # Unary predicate
                entity_type = info["method"].split('(')[1].split(')')[0]
                z3_func = "Function('{}', entity_sorts['{}'], BoolSort())".format(method_name, entity_type)
            elif arity == 2:
                # Binary predicate
                types = info["method"].split('(')[1].split(')')[0].split(', ')
                z3_func = "Function('{}', entity_sorts['{}'], entity_sorts['{}'], BoolSort())".format(method_name, types[0], types[1])

            # Store complete predicate information
            self.predicates[pred_name] = {
                "instance": z3_func,
                "arity": arity,
                "method": info["method"],
                "function": info.get("function", None),  # Optional, may be used for dynamic grounding
            }
        # Print the predicates
        predicates_info = "\n".join(["- {}: {}".format(predicate, details) for predicate, details in self.predicates.items()])
        logger.info("Number of Predicates: {}\nPredicates:\n{}".format(len(self.predicates), predicates_info))

    def _create_rules(self):
        self.rules = {}
        self.rule_tem = {}
        for rule_dict in self.data["Rules"]:
            (rule_name, rule_info), = rule_dict.items()
            # Check if the rule is valid
            formula = rule_info["formula"]
            logger.info("Rule: {} -> \n {}".format(rule_name, formula))

            # Create Z3 variables based on the formula
            var_names = self._extract_variables(formula)
            self.z3_vars = var_names

            # Substitute predicate names in the formula with Z3 function instances
            for method_name, pred_info in self.predicates.items():
                formula = formula.replace(method_name, f'local_predicates["{method_name}"]["instance"]')

            # Now replace the variable names in the formula with their Z3 counterparts
            for var_name in var_names:
                formula = formula.replace(var_name, f'z3_vars["{var_name}"]')

            # Evaluate the modified formula string to create the Z3 expression
            self.rule_tem[rule_name] = formula
        rule_info = "\n".join(["- {}: {}".format(rule, details) for rule, details in self.rule_tem.items()])
        logger.info("Number of Rules: {}\nRules:\n{}".format(len(self.rule_tem), rule_info))
        logger.info("Rules will be grounded later...")

    def _extract_variables(self, formula):
        # Regular expression to find words that start with 'dummy'
        pattern = re.compile(r'\bdummy\w*\b')

        # Find all matches in the formula
        matches = pattern.findall(formula)

        # Remove duplicates by converting the list to a set, then back to a list
        unique_variables = list(set(matches))

        return unique_variables

    def plan(self, world_matrix, intersect_matrix, agents, layerid2listid):
        # 1. Break the global world matrix into local world matrix and split the agents and intersections
        # Note that the local ones will have different size and agent id
        local_world_matrix = world_matrix.clone()
        local_intersections = intersect_matrix.clone()
        s = time.time()
        ego_agent, partial_agents, partial_world, partial_intersections = \
            self.break_world_matrix(local_world_matrix, agents, local_intersections, layerid2listid)
        e = time.time()
        print("Break world matrix time: {}".format(e-s))
        # 2. multi-processing to solve each sub-problem
        combined_results = {}
        # Get the list of keys (agent names) and split them into batches
        agent_keys = list(partial_agents.keys())
        agent_batches = split_into_batches(agent_keys, NUM_PROCESS)
        # Create the pool once
        with Pool(processes=NUM_PROCESS) as pool:
            for batch_keys in agent_batches:
                batch_results = pool.starmap(solve_sub_problem, 
                                            [(ego_name, ego_agent[ego_name].action_mapping, ego_agent[ego_name].action_dist,
                                            self.rule_tem, self.entity_types, self.predicates, self.z3_vars,
                                            partial_agents[ego_name], partial_world[ego_name], partial_intersections[ego_name])
                                            for ego_name in batch_keys])
                
                for result in batch_results:
                    combined_results.update(result)

                # Optional: Manual garbage collection
                gc.collect()
        e2 = time.time()
        print("Solve sub-problem time: {}".format(e2-e))
        return combined_results
    
    def break_world_matrix(self, world_matrix, agents, intersect_matrix, layerid2listid):
        ego_agent = {}
        partial_agents = {}
        partial_world = {}
        partial_intersection = {}
        for agent in agents:
            ego_name = "{}_{}".format(agent.type, agent.layer_id)
            ego_agent[ego_name] = agent
            ego_layer = world_matrix[agent.layer_id]
            ego_position = (ego_layer == TYPE_MAP[agent.type]).nonzero()[0]
            # Calculate the region of the city image that falls within the ego agent's field of view
            x_start = max(ego_position[0]-AGENT_FOV, 0)
            y_start = max(ego_position[1]-AGENT_FOV, 0)
            x_end = min(ego_position[0]+AGENT_FOV+1, world_matrix.shape[1])
            y_end = min(ego_position[1]+AGENT_FOV+1, world_matrix.shape[2])
            partial_world_all = world_matrix[:, x_start:x_end, y_start:y_end].clone()
            partial_intersections = intersect_matrix[:, x_start:x_end, y_start:y_end].clone()
            partial_world_nonzero_int = torch.logical_and(partial_world_all != 0, \
                                                          partial_world_all == partial_world_all.to(torch.int64))
            # Apply torch.any across dimensions 1 and 2 sequentially
            non_zero_layers = partial_world_nonzero_int.any(dim=1).any(dim=1)
            non_zero_layer_indices = torch.where(non_zero_layers)[0]
            partial_world_squeezed = partial_world_all[non_zero_layers]
            partial_world[ego_name] = partial_world_squeezed
            partial_intersection[ego_name] = partial_intersections
            partial_agent = {}
            for layer_id in range(partial_world_squeezed.shape[0]):
                layer = partial_world_squeezed[layer_id]
                layer_nonzero_int = torch.logical_and(layer != 0, layer == layer.to(torch.int64))
                if layer_nonzero_int.nonzero().shape[0] > 1:
                    continue
                non_zero_values = int(layer[layer_nonzero_int.nonzero()[0][0], layer_nonzero_int.nonzero()[0][1]])
                agent_type = LABEL_MAP[non_zero_values]
                # find this agent
                other_agent_layer_id = int(non_zero_layer_indices[layer_id])
                other_agent = agents[layerid2listid[other_agent_layer_id]]
                assert other_agent.type == agent_type
                # ego agent is the first
                if other_agent_layer_id == agent.layer_id:
                    partial_agent["ego"] = PesudoAgent(agent_type, layer_id, other_agent.concepts)
                partial_agent[str(layer_id)] = PesudoAgent(agent_type, layer_id, other_agent.concepts)
            partial_agents[ego_name] = partial_agent
        return ego_agent, partial_agents, partial_world, partial_intersection
            

    def format_rule_string(self, rule_str):
        indent_level = 0
        formatted_str = ""
        bracket_stack = []  # Stack to keep track of brackets

        for char in rule_str:
            if char == ',':
                formatted_str += ',\n' + ' ' * 4 * indent_level
            elif char == '(':
                bracket_stack.append('(')
                formatted_str += '(\n' + ' ' * 4 * (indent_level + 1)
                indent_level += 1
            elif char == ')':
                if not bracket_stack or bracket_stack[-1] != '(':
                    raise ValueError("Unmatched closing bracket detected.")
                bracket_stack.pop()
                indent_level -= 1
                formatted_str += '\n' + ' ' * 4 * indent_level + ')'
            else:
                formatted_str += char

        if bracket_stack:
            raise ValueError("Unmatched opening bracket detected.")

        return formatted_str

def solve_sub_problem(ego_name, 
                      ego_action_mapping,
                      ego_action_dist,
                      rule_tem, 
                      entity_types, 
                      predicates, 
                      var_names,
                      partial_agents, 
                      partial_world, 
                      partial_intersections):
    # 1. create solver
    local_solver = Solver()
    # 2. create sorts and variables
    entity_sorts = {}
    for entity_type in entity_types:
        entity_sorts[entity_type] = DeclareSort(entity_type)
    z3_vars = {var_name: Const(var_name, entity_sorts[var_name.replace('dummy', '')]) \
                       for var_name in var_names}
    # 3. partial world to entities
    local_entities = world2entity(entity_sorts, partial_intersections, partial_agents)
    # 4. create, ground predicates and add to solver
    local_predicates = copy.deepcopy(predicates)
    for pred_name, pred_info in local_predicates.items():
        eval_pred = eval(pred_info["instance"])
        pred_info["instance"] = eval_pred
        arity = pred_info["arity"]

        # Import the grounding method
        method_full_name = pred_info["function"]
        if method_full_name == "None":
            continue
        module_name, method_name = method_full_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        method = getattr(module, method_name)

        if arity == 1:
            # Unary predicate grounding
            for entity in local_entities[eval_pred.domain(0).name()]:
                entity_name = entity.decl().name()
                value = method(partial_world, partial_intersections, partial_agents, entity_name)
                if value:
                    local_solver.add(eval_pred(entity))
                else:
                    local_solver.add(Not(eval_pred(entity)))
        elif arity == 2:
            # Binary predicate grounding
            for entity1 in local_entities[eval_pred.domain(0).name()]:
                entity1_name = entity1.decl().name()
                for entity2 in local_entities[eval_pred.domain(1).name()]:
                    entity2_name = entity2.decl().name()
                    value = method(partial_world, partial_intersections, partial_agents, entity1_name, entity2_name)
                    if value:
                        local_solver.add(eval_pred(entity1, entity2))
                    else:
                        local_solver.add(Not(eval_pred(entity1, entity2)))

    # 5. create, ground rules and add to solver
    local_rule_tem = copy.deepcopy(rule_tem)
    for rule_name, rule_template in local_rule_tem.items():
        # the first entity is the ego agent
        agent = local_entities["Agent"][0]
        # Replace placeholder in the rule template with the actual agent entity
        instantiated_rule = eval(rule_template)
        local_solver.add(instantiated_rule)

    # **Important: Closed world quantifier rule, to ensure z3 do not add new entity to satisfy the rule and "dummy" is not part of the world**
    for var_name, z3_var in z3_vars.items():
        entity_list = local_entities[var_name.replace('dummy', '')]
        constraint = Or([z3_var == entity for entity in entity_list])
        local_solver.add(ForAll([z3_var], constraint))
    
    # 6. solve
    if local_solver.check() == sat:
        model = local_solver.model()
        # Interpret the solution to the FOL problem
        action_mapping = ego_action_mapping
        action_dist = torch.zeros_like(ego_action_dist)

        for key in local_predicates.keys():
            action = []
            for action_id, action_name in action_mapping.items():
                if key in action_name:
                    action.append(action_id)
            if len(action)>0:
                for a in action:
                    if is_true(model.evaluate(local_predicates[key]["instance"](local_entities["Agent"][0]))):
                        action_dist[a] = 1.0
        # No action specified, use the default action, Normal
        if action_dist.sum() == 0:
            for action_id, action_name in action_mapping.items():
                if "Normal" in action_name:
                    action_dist[action_id] = 1.0

        agents_actions = {ego_name: action_dist}
        return agents_actions
    else:
        # No solution means do not exist intersection/agent in the field of view, Normal
        # Interpret the solution to the FOL problem
        action_mapping = ego_action_mapping
        action_dist = torch.zeros_like(ego_action_dist)

        for action_id, action_name in action_mapping.items():
            if "Normal" in action_name:
                action_dist[action_id] = 1.0

        agents_actions = {ego_name: action_dist}
        return agents_actions

def split_into_batches(keys, batch_size):
    """Split keys into batches of a given size."""
    for i in range(0, len(keys), batch_size):
        yield keys[i:i + batch_size]

def world2entity(entity_sorts, partial_intersect, partial_agents):
    assert "Agent" in entity_sorts.keys() and "Intersection" in entity_sorts.keys()
    # all the enitities are stored in self.entities
    entities = {}
    for entity_type in entity_sorts.keys():
        entities[entity_type] = []
        # For Agents
        if entity_type == "Agent":
            for key, agent in partial_agents.items():
                if key == "ego":
                    continue
                agent_id = agent.layer_id
                agent_type = agent.type
                agent_name = f"Agent_{agent_type}_{agent_id}"
                # Create a Z3 constant for the agent
                agent_entity = Const(agent_name, entity_sorts['Agent'])
                entities[entity_type].append(agent_entity)
        elif entity_type == "Intersection":
            # For Intersections
            unique_intersections = np.unique(partial_intersect[0])
            unique_intersections = unique_intersections[unique_intersections != 0]
            for intersection_id in unique_intersections:
                intersection_name = f"Intersection_{intersection_id}"
                # Create a Z3 constant for the intersection
                intersection_entity = Const(intersection_name, entity_sorts['Intersection'])
                entities[entity_type].append(intersection_entity)
    return entities
