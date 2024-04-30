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
from .z3_rl import world2entity, get_action_name, logic_grounding_shape
from .z3_expert import Z3PlannerExpert

logger = logging.getLogger(__name__)

class Z3PlannerExpertES(Z3PlannerExpert):
    def __init__(self, yaml_path):        
        super().__init__(yaml_path)
        self.rl_input_shape = None
        self.last_rl_obs = None
        self.max_priority = None

    def reset(self):
        self.last_rl_obs = None
        self.max_priority = None

    def plan(self, world_matrix, 
             intersect_matrix, 
             agents, 
             layerid2listid, 
             use_multiprocessing=True, 
             rl_agent=None):
        # 1. Break the global world matrix into local world matrix and split the agents and intersections
        # Note that the local ones will have different size and agent id
        # e = time.time()
        local_world_matrix = world_matrix.clone()
        local_intersections = intersect_matrix.clone()
        ego_agent, partial_agents, partial_world, partial_intersections, rl_flags = \
            self.break_world_matrix(local_world_matrix, agents, local_intersections, layerid2listid, rl_agent)
        # logger.info("Break world time: {}".format(time.time()-e))
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
                                                partial_agents[ego_name], partial_world[ego_name], partial_intersections[ego_name],
                                                self.fov_entities, rl_flags[ego_name], self.rl_input_shape)
                                                for ego_name in batch_keys])
                    
                    for result in batch_results:
                        combined_results.update(result)
                    gc.collect()
        else:
            # Looping approach
            for ego_name in agent_keys:
                if rl_flags[ego_name]:
                    # RL agent only gets the observation
                    result = solve_sub_problem(ego_name, ego_agent[ego_name].action_mapping, ego_agent[ego_name].action_dist,
                                            self.rules['Expert'], self.entity_types, self.predicates, self.z3_vars,
                                            partial_agents[ego_name], partial_world[ego_name], partial_intersections[ego_name], 
                                            self.fov_entities, True, rl_input_shape=self.rl_input_shape, semantic_pred2index=self.semantic_pred2index, \
                                            max_priority=self.max_priority)
                    self.last_rl_obs = {
                        "last_obs_dict": copy.deepcopy(result["{}_grounding_dic".format(ego_name)]),
                        "last_obs": result["{}_grounding".format(ego_name)].copy(),
                        "expert_action": result["{}_action".format(ego_name)].clone(),
                    }
                else:
                    result = solve_sub_problem(ego_name, ego_agent[ego_name].action_mapping, ego_agent[ego_name].action_dist,
                                            self.rules['Sim'], self.entity_types, self.predicates, self.z3_vars,
                                            partial_agents[ego_name], partial_world[ego_name], partial_intersections[ego_name], 
                                            self.fov_entities, False)
                combined_results.update(result)

        e2 = time.time()
        # logger.info("Solve sub-problem time: {}".format(e2-e))
        return combined_results
    
    def eval(self, rl_action):
        if self.last_rl_obs is None:
            return 0
        fail, reward = eval_action(rl_action, self.rules['Task'], self.entity_types, self.predicates, self.z3_vars, self.fov_entities,
                             self.last_rl_obs["last_obs_dict"], self.last_rl_obs["last_obs"])
        self.last_rl_obs = None
        return fail, reward
    
    def eval_state_action(self, state, action):
        """
        Evaluate the state-action pair
        state: the current grounding of the world, the same shape and type as: self.last_rl_obs["last_obs"]
        action: the action to be taken
        """
        # 1. conver the state to the last_rl_obs dict format
        last_obs_dict = self.grounding2dict(state)
        # 2. evaluate the action similar to the eval method
        fail, reward = eval_action(action, self.rules['Task'], self.entity_types, self.predicates, self.z3_vars, self.fov_entities,
                                last_obs_dict, state)
        del last_obs_dict
        return fail, reward

    def grounding2dict(self, grounding):
        """
        Convert the grounding to the last_rl_obs dict format
        """
        last_obs_dict = {}
        for keys, value in self.pred_grounding_index.items():
            s, e = value[0], value[1]
            for i in range(0, e-s):
                name = keys + "_{}".format(i)
                last_obs_dict[name] = grounding[s+i]
        return last_obs_dict

    def logic_grounding_shape(self, fov_entities):
            self.fov_entities = fov_entities
            lossy_input_shape, pred_grounding_ind = logic_grounding_shape(self.entity_types, self.predicates, self.z3_vars, fov_entities)
            self.pred_grounding_index = pred_grounding_ind
            return lossy_input_shape, pred_grounding_ind
    
    def init_es_input_shape(self, es_input_shape, pred2idex):
        self.rl_input_shape = es_input_shape
        self.semantic_pred2index = pred2idex
        max_index = max(pred2idex.values())
        self.semantic_pred2index["direction"] = max_index + 1
        self.semantic_pred2index["dx"] = self.semantic_pred2index["direction"] + 4
        self.semantic_pred2index["dy"] = self.semantic_pred2index["dx"] + 1
        self.semantic_pred2index["priority"] = self.semantic_pred2index["dy"] + 1

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
                      fov_entities,
                      rl_flag,
                      rl_input_shape=None,
                      semantic_pred2index=None,
                      max_priority=None):
    grounding = []
    grounding_dic = {}
    fov = (AGENT_FOV + 1) * (2 * AGENT_FOV + 1)
    ego_pos = torch.zeros(2, dtype=torch.float32) - 1
    obs_array_es = torch.zeros(fov * rl_input_shape, dtype=torch.float32)
    obs_matrix_es = torch.zeros((rl_input_shape, partial_intersections.shape[1], partial_intersections.shape[2]), dtype=torch.float32)
    # 0-3 is intersection matrix
    obs_matrix_es[:3] =  partial_intersections > 0
    # 1. create sorts and variables
    entity_sorts = {}
    for entity_type in entity_types:
        entity_sorts[entity_type] = DeclareSort(entity_type)
    z3_vars = {var_name: Const(var_name, entity_sorts['Entity']) \
                       for var_name in var_names}
    # 2. partial world to entities
    local_entities = world2entity(entity_sorts, partial_intersections, partial_agents, fov_entities, rl_flag)
    # 3. create, ground predicates and add to solver
    local_predicates = copy.deepcopy(predicates)
    # 4. create, ground predicates and add to solver
    if not rl_flag:
        local_solver = Solver()
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
                    for a in action:
                        if is_true(model.evaluate(local_predicates[key]["instance"](local_entities["Entity"][0]))):
                            action_dist[a] = 1.0
            # No action specified, use the default action, Normal
            if action_dist.sum() == 0:
                for action_id, action_name in action_mapping.items():
                    if "Normal" in action_name:
                        action_dist[action_id] = 1.0

            agents_actions = {ego_name: action_dist}
            return agents_actions
        else:
            action_mapping = ego_action_mapping
            action_dist = torch.zeros_like(ego_action_dist)

            for action_id, action_name in action_mapping.items():
                if "Normal" in action_name:
                    action_dist[action_id] = 1.0

            agents_actions = {ego_name: action_dist}
            return agents_actions
    else:
        # SAT solver helps build an expert
        scene_graph = {
        'width': 0,
        'height': 0,
        'objects': {}
        }
        for ent in local_entities["Entity"]:
            entity_name = ent.decl().name()
            _, obj_name, layer_id = entity_name.split("_")
            scene_graph["objects"][layer_id] = {
                'name': obj_name,
                'h': 0,
                'w': 0,
                'x': 0,
                'y': 0,
                'relations': [],
                'attributes': []
            }
        local_solver = Solver()
        for pred_name, pred_info in local_predicates.items():
            k = 0
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
                    _, agent_type, layer_id = entity_name.split("_")
                    value = method(partial_world, partial_intersections, partial_agents, entity_name)
                    if value:
                        scene_graph["objects"][layer_id]["attributes"].append(pred_name)
                        local_solver.add(eval_pred(entity))
                        grounding_dic["{}_{}".format(pred_name, k)] = 1
                        grounding.append(1)
                        k += 1
                        if pred_name in semantic_pred2index.keys():
                            # add to obs matrix
                            layer_id_int = int(layer_id)
                            agent_layer = partial_world[layer_id_int]
                            agent_position = (agent_layer == TYPE_MAP[agent_type]).nonzero()[0]
                            # semantic
                            obs_matrix_es[semantic_pred2index[pred_name], agent_position[0], agent_position[1]] = 1
                            # direction and priority
                            if layer_id in partial_agents.keys():
                                priority = partial_agents[layer_id].priority
                                direction = partial_agents[layer_id].moving_direction
                            else:
                                assert "ego_{}".format(layer_id) in partial_agents.keys()
                                direction = partial_agents["ego_{}".format(layer_id)].moving_direction
                                priority = partial_agents["ego_{}".format(layer_id)].priority
                                ego_pos = agent_position
                            if direction is not None:
                                one_hot = direction2onehot(direction)
                                obs_matrix_es[semantic_pred2index["direction"]:semantic_pred2index["direction"]+4, \
                                              agent_position[0], agent_position[1]] = one_hot
                            obs_matrix_es[semantic_pred2index["priority"], agent_position[0], agent_position[1]] = priority/max_priority
                    else:
                        local_solver.add(Not(eval_pred(entity)))
                        grounding_dic["{}_{}".format(pred_name, k)] = 0
                        grounding.append(0)
                        k += 1
            elif arity == 2:
                # Binary predicate grounding
                for entity1 in local_entities[eval_pred.domain(0).name()]:
                    entity1_name = entity1.decl().name()
                    _, _, layer_id1 = entity1_name.split("_")
                    for entity2 in local_entities[eval_pred.domain(1).name()]:
                        entity2_name = entity2.decl().name()
                        _, _, layer_id2 = entity2_name.split("_")
                        value = method(partial_world, partial_intersections, partial_agents, entity1_name, entity2_name)
                        if value:
                            relation_info = {
                                "name": pred_name,
                                "object": layer_id2
                            }
                            scene_graph["objects"][layer_id1]["relations"].append(relation_info)
                            local_solver.add(eval_pred(entity1, entity2))
                            grounding_dic["{}_{}".format(pred_name, k)] = 1
                            grounding.append(1)
                            k += 1
                        else:
                            local_solver.add(Not(eval_pred(entity1, entity2)))
                            grounding_dic["{}_{}".format(pred_name, k)] = 0
                            grounding.append(0)
                            k += 1

        # 5. create, ground rules and add to solver
        local_rule_tem = copy.deepcopy(rule_tem)
        for rule_name, rule_template in local_rule_tem.items():
            # the first entity is the ego agent
            entity = local_entities["Entity"][0]
            # Replace placeholder in the rule template with the actual agent entity
            instantiated_rule = eval(rule_template["content"])
            local_solver.add(instantiated_rule)

        # **Important: Closed world quantifier rule, to ensure z3 do not add new entity to satisfy the rule and "dummy" is not part of the world**
        for var_name, z3_var in z3_vars.items():
            entity_list = local_entities["Entity"]
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
                        if is_true(model.evaluate(local_predicates[key]["instance"](local_entities["Entity"][0]))):
                            action_dist[a] = 1.0
                            _, _, layer_id = local_entities["Entity"][0].decl().name().split("_")
                            if key not in scene_graph["objects"][layer_id]["attributes"]:
                                scene_graph["objects"][layer_id]["attributes"].append(key)
            # No action specified, use the default action, Normal
            if action_dist.sum() == 0:
                for action_id, action_name in action_mapping.items():
                    if "Normal" in action_name:
                        action_dist[action_id] = 1.0

        # 7. comput dx dy for the obs matrix
        assert ego_pos[0] != -1, "Ego position is not found"
        # Get the shape of the observation matrix
        rows, cols = obs_matrix_es.shape[1], obs_matrix_es.shape[2]

        # Generate a range of indices
        i_indices = torch.arange(rows).reshape(rows, 1)  # Column vector
        j_indices = torch.arange(cols).reshape(1, cols)  # Row vector

        # Calculate dx and dy using broadcasting
        dx = (i_indices - ego_pos[0]) / rows
        dy = (j_indices - ego_pos[1]) / cols

        # Assign these calculations back to obs_matrix_es
        obs_matrix_es[semantic_pred2index["dx"], :, :] = dx
        obs_matrix_es[semantic_pred2index["dy"], :, :] = dy
        # from matrix to array
        flatten_obs = obs_matrix_es.flatten()
        if flatten_obs.shape[0] <= obs_array_es.shape[0]:
            obs_array_es[:flatten_obs.shape[0]] = flatten_obs
        elif flatten_obs.shape[0] > obs_array_es.shape[0]:
            # if the obs is too large, then only take the first part
            obs_array_es = obs_matrix_es[:obs_array_es.shape[0]]
        
        agents_actions = {
            "{}_action".format(ego_name): action_dist,
            "{}_grounding".format(ego_name): np.array(grounding, dtype=np.float32),
            "{}_grounding_dic".format(ego_name): grounding_dic,
            "{}_scene_graph".format(ego_name): scene_graph,
            "{}_obs_es".format(ego_name): obs_array_es
        }
        assert len(grounding) == rl_input_shape

        return agents_actions

def eval_action(rl_action,
                rule_tem, 
                entity_types, 
                predicates, 
                var_names,
                fov_entities,
                last_obs_dict,
                last_obs):
    grounding = []
    # 1. create sorts and variables
    entity_sorts = {}
    for entity_type in entity_types:
        entity_sorts[entity_type] = DeclareSort(entity_type)
    z3_vars = {var_name: Const(var_name, entity_sorts["Entity"]) \
                       for var_name in var_names}
    # 2. entities
    entities = {}
    for entity_type in entity_sorts.keys():
        entity_num = fov_entities[entity_type]
        entities[entity_type] = [Const(f"{entity_type}_{i}", entity_sorts[entity_type]) for i in range(entity_num)]
    # 3. create, ground predicates and add to solver
    local_predicates = copy.deepcopy(predicates)
    # 4. create, ground predicates and add to solver
    local_solvers = {rule_name: Solver() for rule_name in rule_tem.keys()}
    for pred_name, pred_info in local_predicates.items():
        k = 0
        eval_pred = eval(pred_info["instance"])
        pred_info["instance"] = eval_pred
        arity = pred_info["arity"]

        # Import the grounding method
        method_full_name = pred_info["function"]

        if method_full_name == "None":
            assert rl_action is not None, "Make sure the rl_action is not None"
            # ego action
            action_name = get_action_name(rl_action)
            if pred_name == action_name:
                for rule_name, rule_template in rule_tem.items():
                    local_solvers[rule_name].add(eval_pred(entities["Entity"][0]))
            else:
                for rule_name, rule_template in rule_tem.items():
                    local_solvers[rule_name].add(Not(eval_pred(entities["Entity"][0])))
            continue

        if arity == 1:
            # Unary predicate grounding
            for entity in entities[eval_pred.domain(0).name()]:
                if last_obs_dict["{}_{}".format(pred_name, k)]:
                    grounding.append(1)
                    k += 1
                    for rule_name, rule_template in rule_tem.items():
                        local_solvers[rule_name].add(eval_pred(entity))
                else:
                    grounding.append(0)
                    k += 1
                    for rule_name, rule_template in rule_tem.items():
                        local_solvers[rule_name].add(Not(eval_pred(entity)))
        elif arity == 2:
            # Binary predicate grounding
            for entity1 in entities[eval_pred.domain(0).name()]:
                for entity2 in entities[eval_pred.domain(1).name()]:
                    if last_obs_dict["{}_{}".format(pred_name, k)]:
                        grounding.append(1)
                        k += 1
                        for rule_name, rule_template in rule_tem.items():
                            local_solvers[rule_name].add(eval_pred(entity1, entity2))
                    else:
                        grounding.append(0)
                        k += 1
                        for rule_name, rule_template in rule_tem.items():
                            local_solvers[rule_name].add(Not(eval_pred(entity1, entity2)))

    # 5. create, ground rules and add to solver
    local_rule_tem = copy.deepcopy(rule_tem)
    for rule_name, rule_template in local_rule_tem.items():
        # the first entity is the ego agent
        entity = entities["Entity"][0]
        # Replace placeholder in the rule template with the actual agent entity
        instantiated_rule = eval(rule_template["content"])
        local_solvers[rule_name].add(instantiated_rule)

    # **Important: Closed world quantifier rule, to ensure z3 do not add new entity to satisfy the rule and "dummy" is not part of the world**
    for var_name, z3_var in z3_vars.items():
        entity_list = entities['Entity']
        for rule_name, rule_template in rule_tem.items():
            constraint = Or([z3_var == entity for entity in entity_list])
            local_solvers[rule_name].add(ForAll([z3_var], constraint))
    
    # 6. solve for reward
    obs = np.array(grounding, dtype=np.float32)
    assert np.all(obs == last_obs), print(obs, last_obs)
    fail = False
    reward = 0
    for rule_name, rule_solver in local_solvers.items():
        if rule_solver.check() == sat:
                continue
        else:
            if rule_tem[rule_name]["dead"]:
                fail = True
            reward += local_rule_tem[rule_name]["reward"]

    # When really use the expert policy, enable this checking
    # assert not fail, "Expert never obeys rules"
    return fail, reward

def direction2onehot(direction):
    if direction == "Left":
        return torch.tensor([1, 0, 0, 0], dtype=torch.float32)
    elif direction == "Right":
        return torch.tensor([0, 1, 0, 0], dtype=torch.float32)
    elif direction == "Up":
        return torch.tensor([0, 0, 1, 0], dtype=torch.float32)
    elif direction == "Down":
        return torch.tensor([0, 0, 0, 1], dtype=torch.float32)
    else:
        raise ValueError("Invalid direction")
