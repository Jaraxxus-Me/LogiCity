import re
import gc
import time
import copy
import torch
import logging
import importlib
import numpy as np
from z3 import *
from ...core.config import *
from multiprocessing import Pool
from ...utils.find import find_agent
from ...utils.sample import split_into_subsets
from .basic import LocalPlanner

logger = logging.getLogger(__name__)

# used for grounding
class PesudoAgent:
    def __init__(self, type, layer_id, concepts, moving_direction):
        self.type = type
        self.layer_id = layer_id
        self.type = concepts["type"]
        self.priority = concepts["priority"]
        self.concepts = concepts
        self.moving_direction = moving_direction

class Z3Planner(LocalPlanner):
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

    def plan(self, world_matrix, intersect_matrix, agents, layerid2listid, use_multiprocessing=True):
        # 1. Break the global world matrix into local world matrix and split the agents and intersections
        # Note that the local ones will have different size and agent id
        e = time.time()
        local_world_matrix = world_matrix.clone()
        local_intersections = intersect_matrix.clone()
        ego_agent, partial_agents, partial_world, partial_intersections, local2global_layer_id_map = \
            self.break_world_matrix(local_world_matrix, agents, local_intersections, layerid2listid)
        logger.info("Break world time: {}".format(time.time()-e))
        # 2. Choose between multi-processing and looping
        combined_results = {}
        agent_keys = list(partial_agents.keys())
        
        if use_multiprocessing:
            # Multi-processing approach
            agent_batches = split_into_batches(agent_keys, NUM_PROCESS)
            with Pool(processes=NUM_PROCESS) as pool:
                for batch_keys in agent_batches:
                    batch_results = pool.starmap(solve_sub_problem, 
                                                [(ego_name, ego_agent[ego_name].action_mapping, ego_agent[ego_name].action_dist,
                                                self.rule_tem, self.entity_types, self.predicates, self.z3_vars,
                                                partial_agents[ego_name], partial_world[ego_name], partial_intersections[ego_name])
                                                for ego_name in batch_keys])
                    
                    for result in batch_results:
                        combined_results.update(result)
                    gc.collect()
        else:
            # Looping approach
            for ego_name in agent_keys:
                result = solve_sub_problem(ego_name, ego_agent[ego_name].action_mapping, ego_agent[ego_name].action_dist,
                                        self.rule_tem, self.entity_types, self.predicates, self.z3_vars,
                                        partial_agents[ego_name], partial_world[ego_name], partial_intersections[ego_name],
                                        local2global_layer_id_map[ego_name])
                combined_results.update(result)

        e2 = time.time()
        logger.info("Solve sub-problem time: {}".format(e2-e))
        return combined_results
    
    def get_fov(self, position, direction, width, height):
        # Calculate the region of the city image that falls within the ego agent's field of view
        if direction == None:
            x_start = max(position[0]-AGENT_FOV, 0)
            y_start = max(position[1]-AGENT_FOV, 0)
            x_end = min(position[0]+AGENT_FOV+1, width)
            y_end = min(position[1]+AGENT_FOV+1, height)
        elif direction == "Left":
            x_start = max(position[0]-AGENT_FOV, 0)
            y_start = max(position[1]-AGENT_FOV, 0)
            x_end = min(position[0]+AGENT_FOV+1, width)
            y_end = min(position[1]+2, height)
        elif direction == "Right":
            x_start = max(position[0]-AGENT_FOV, 0)
            y_start = max(position[1]-2, 0)
            x_end = min(position[0]+AGENT_FOV+1, width)
            y_end = min(position[1]+AGENT_FOV+1, height)
        elif direction == "Up":
            x_start = max(position[0]-AGENT_FOV, 0)
            y_start = max(position[1]-AGENT_FOV, 0)
            x_end = min(position[0]+2, width)
            y_end = min(position[1]+AGENT_FOV+1, height)
        elif direction == "Down":
            x_start = max(position[0]-2, 0)
            y_start = max(position[1]-AGENT_FOV, 0)
            x_end = min(position[0]+AGENT_FOV+1, width)
            y_end = min(position[1]+AGENT_FOV+1, height)
        return x_start, y_start, x_end, y_end

    def break_world_matrix(self, world_matrix, agents, intersect_matrix, layerid2listid):
        ego_agent = {}
        partial_agents = {}
        partial_world = {}
        partial_intersection = {}
        local2global_layer_id_map = {}
        for agent in agents:
            ego_name = "{}_{}".format(agent.type, agent.layer_id)
            ego_agent[ego_name] = agent
            ego_layer = world_matrix[agent.layer_id]
            ego_position = (ego_layer == TYPE_MAP[agent.type]).nonzero()[0]
            ego_direction = agent.last_move_dir
            x_start, y_start, x_end, y_end = self.get_fov(ego_position, ego_direction, world_matrix.shape[1], world_matrix.shape[2])
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
            local2global_layer_id_map[ego_name] = {}
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
                if other_agent_layer_id == agent.layer_id:
                    partial_agent["ego_{}".format(layer_id)] = PesudoAgent(agent_type, layer_id, other_agent.concepts, other_agent.last_move_dir)
                else:
                    partial_agent[str(layer_id)] = PesudoAgent(agent_type, layer_id, other_agent.concepts, other_agent.last_move_dir)
                local2global_layer_id_map[ego_name][layer_id] = non_zero_layer_indices[layer_id]
            partial_agents[ego_name] = partial_agent
        return ego_agent, partial_agents, partial_world, partial_intersection, local2global_layer_id_map

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
                      partial_intersections,
                      local2global_layer_id_map):
    grounding_dic = {}
    # 1. create solver
    local_solver = Solver()
    # 2. create sorts and variables
    entity_sorts = {}
    for entity_type in entity_types:
        entity_sorts[entity_type] = DeclareSort(entity_type)
    z3_vars = {var_name: Const(var_name, entity_sorts['Entity']) \
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
                global_layer_id = local2global_layer_id_map[int(entity_name.split('_')[-1])]
                grounding_dic["{}_{}".format(pred_name, global_layer_id)] = bool(value)
                if value:
                    local_solver.add(eval_pred(entity))
                else:
                    local_solver.add(Not(eval_pred(entity)))
        elif arity == 2:
            # Binary predicate grounding
            for entity1 in local_entities[eval_pred.domain(0).name()]:
                entity1_name = entity1.decl().name()
                global_layer_id_1 = local2global_layer_id_map[int(entity1_name.split('_')[-1])]
                for entity2 in local_entities[eval_pred.domain(1).name()]:
                    entity2_name = entity2.decl().name()
                    value = method(partial_world, partial_intersections, partial_agents, entity1_name, entity2_name)
                    global_layer_id_2 = local2global_layer_id_map[int(entity2_name.split('_')[-1])]
                    grounding_dic["{}_{}_{}".format(pred_name, global_layer_id_1, global_layer_id_2)] = bool(value)
                    if value:
                        local_solver.add(eval_pred(entity1, entity2))
                    else:
                        local_solver.add(Not(eval_pred(entity1, entity2)))

    # 5. create, ground rules and add to solver
    local_rule_tem = copy.deepcopy(rule_tem)
    for rule_name, rule_template in local_rule_tem.items():
        # the first entity is the ego agent
        entity = local_entities["Entity"][0]
        # Replace placeholder in the rule template with the actual agent entity
        instantiated_rule = eval(rule_template)
        local_solver.add(instantiated_rule)

    # **Important: Closed world quantifier rule, to ensure z3 do not add new entity to satisfy the rule and "dummy" is not part of the world**
    for var_name, z3_var in z3_vars.items():
        entity_list = local_entities['Entity']
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
                if is_true(model.evaluate(local_predicates[key]["instance"](local_entities["Entity"][0]))):
                    for a in action:
                        action_dist[a] = 1.0
        # No action specified, use the default action, Normal
        if action_dist.sum() == 0:
            for action_id, action_name in action_mapping.items():
                if "Normal" in action_name:
                    action_dist[action_id] = 1.0
    else:
        # No solution means do not exist intersection/agent in the field of view, Normal
        # Interpret the solution to the FOL problem
        action_mapping = ego_action_mapping
        action_dist = torch.zeros_like(ego_action_dist)

        for action_id, action_name in action_mapping.items():
            if "Normal" in action_name:
                action_dist[action_id] = 1.0

    agents_actions = {ego_name: action_dist, "{}_grounding_dic".format(ego_name): grounding_dic}
    return agents_actions

def split_into_batches(keys, batch_size):
    """Split keys into batches of a given size."""
    for i in range(0, len(keys), batch_size):
        yield keys[i:i + batch_size]

def world2entity(entity_sorts, partial_intersect, partial_agents):
    # all the enitities are stored in self.entities
    entities = {}
    for entity_type in entity_sorts.keys():
        entities[entity_type] = []
        # For Agents
        if entity_type == "Entity":
            # all entities are agents
            for key, agent in partial_agents.items():
                if "ego" in key:
                    ego_agent = agent
                    continue
                agent_id = agent.layer_id
                agent_type = agent.type
                agent_name = f"Entity_{agent_type}_{agent_id}"
                # Create a Z3 constant for the agent
                agent_entity = Const(agent_name, entity_sorts[entity_type])
                entities[entity_type].append(agent_entity)
            agent_id = ego_agent.layer_id
            agent_type = ego_agent.type
            agent_name = f"Entity_{agent_type}_{agent_id}"
            # Create a Z3 constant for the agent
            agent_entity = Const(agent_name, entity_sorts[entity_type])
            # ego agent is the first
            entities[entity_type] = [agent_entity] + entities[entity_type]
    return entities
