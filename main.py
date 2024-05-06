import os
import copy
import time
import yaml
import torch
import argparse
import importlib
import numpy as np
import pickle as pkl
from tqdm import trange
from logicity.core.config import *
from logicity.utils.load import CityLoader
from logicity.utils.logger import setup_logger
from logicity.utils.vis import *
# RL
from logicity.rl_agent.alg import *
from logicity.utils.gym_wrapper import GymCityWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv
from logicity.utils.gym_callback import EvalCheckpointCallback

def parse_arguments():
    parser = argparse.ArgumentParser(description='Logic-based city simulation.')
    # logger
    parser.add_argument('--log_dir', type=str, default="./log_sim")
    parser.add_argument('--exp', type=str, default="sim_easy2")
    parser.add_argument('--vis', action='store_true', help='Visualize the city.')
    # seed
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--max_steps', type=int, default=200)
    # RL
    parser.add_argument('--collect_only', action='store_true', help='Only collect expert data.')
    parser.add_argument('--use_gym', action='store_true', help='In gym mode, we can use RL alg. to control certain agents.')
    parser.add_argument('--save_steps', action='store_true', help='Save step-wise decision for each trajectory.')
    parser.add_argument('--config', default=None, help='Configure file for this RL exp.')
    parser.add_argument('--checkpoint_path', default=None, help='Path to the trained model.')
    # vis_dataset
    parser.add_argument('--create_vis_dataset', action='store_true', help='Create vis dataset from scratch.')
    parser.add_argument("--mode", type=str, default='easy')
    parser.add_argument('--train_world_num', type=int, default=20)
    parser.add_argument('--val_world_num', type=int, default=5)
    parser.add_argument('--test_world_num', type=int, default=5)
    parser.add_argument('--min_agent_num_train', type=int, default=5)
    parser.add_argument('--max_agent_num_train', type=int, default=8)
    parser.add_argument('--min_agent_num_val', type=int, default=5)
    parser.add_argument('--max_agent_num_val', type=int, default=8)
    parser.add_argument('--min_agent_num_test', type=int, default=5)
    parser.add_argument('--max_agent_num_test', type=int, default=8)
    parser.add_argument('--dataset_dir', type=str, default="./vis_dataset/easy_200")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def dynamic_import(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def make_env(simulation_config, episode_cache=None, return_cache=False): 
    # Unpack arguments from simulation_config and pass them to CityLoader
    city, cached_observation = CityLoader.from_yaml(**simulation_config, episode_cache=episode_cache)
    env = GymCityWrapper(city)
    if return_cache: 
        return env, cached_observation
    else:
        return env
    
def make_envs(simulation_config, rank):
    """
    Utility function for multiprocessed env.
    
    :param simulation_config: The configuration for the simulation.
    :param rank: Unique index for each environment to ensure different seeds.
    :return: A function that creates a single environment.
    """
    def _init():
        env = make_env(simulation_config)
        env.seed(rank + 1000)  # Optional: set a unique seed for each environment
        return env
    return _init

def main_collect(args, logger):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = load_config(args.config)
    simulation_config = config["simulation"]
    logger.info("Simulation config: {}".format(simulation_config))
    collection_config = config['collecting_config']
    logger.info("RL config: {}".format(collection_config))

    # Check if expert data collection is requested
    logger.info("Collecting expert demonstration data...")
    # Create an environment instance for collecting expert demonstrations
    expert_data_env, cached_observation = make_env(simulation_config, None, True)  # Use your existing environment setup function
    assert expert_data_env.use_expert  # Ensure the environment uses expert actions
    
    # Initialize the ExpertCollector with the environment and total timesteps
    collector = ExpertCollector(expert_data_env, **collection_config)
    _, full_world = collector.collect_data(cached_observation)
    
    # Save the collected expert demonstrations
    collector.save_data(f"{args.log_dir}/{args.exp}_expert_demonstrations.pkl")
    logger.info(f"Collected and saved expert demonstration data to {args.log_dir}/{args.exp}_expert_demonstrations.pkl")
    # Save the full world if needed
    if collection_config["return_full_world"]:
        for ts in range(len(full_world)):
            with open(os.path.join(args.log_dir, "{}_{}.pkl".format(args.exp, ts)), "wb") as f:
                pkl.dump(full_world[ts], f)

def main(args, logger):
    config = load_config(args.config)
    # simulation config
    simulation_config = config["simulation"]
    logger.info("Simulation config: {}".format(simulation_config))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Create a city instance with a predefined grid
    city, cached_observation = CityLoader.from_yaml(**simulation_config)
    visualize_city(city, 4*WORLD_SIZE, -1, "vis/init.png")
    # Main simulation loop
    steps = 0
    while steps < args.max_steps:
        logger.info("Simulating Step_{}...".format(steps))
        s = time.time()
        time_obs = city.update()
        e = time.time()
        logger.info("Time spent: {}".format(e-s))
        # Visualize the current state of the city (optional)
        if args.vis:
            visualize_city(city, 4*WORLD_SIZE, -1, "vis/step_{}.png".format(steps))
        steps += 1
        cached_observation["Time_Obs"][steps] = time_obs

    # Save the cached observation for better rendering
    with open(os.path.join(args.log_dir, "{}.pkl".format(args.exp)), "wb") as f:
        pkl.dump(cached_observation, f)

def main_gym(args, logger): 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = load_config(args.config)
    # simulation config
    simulation_config = config["simulation"]
    logger.info("Simulation config: {}".format(simulation_config))
    # RL config
    rl_config = config['stable_baselines']
    logger.info("RL config: {}".format(rl_config))
    # Dynamic import of the features extractor class
    if "features_extractor_module" in rl_config["policy_kwargs"]:
        features_extractor_class = dynamic_import(
            rl_config["policy_kwargs"]["features_extractor_module"],
            rl_config["policy_kwargs"]["features_extractor_class"]
        )
        # Prepare policy_kwargs with the dynamically imported class
        policy_kwargs = {
            "features_extractor_class": features_extractor_class,
            "features_extractor_kwargs": rl_config["policy_kwargs"]["features_extractor_kwargs"]
        }
    else:
        policy_kwargs = rl_config["policy_kwargs"]
    # Dynamic import of the RL algorithm
    algorithm_class = dynamic_import(
        "logicity.rl_agent.alg",  # Adjust the module path as needed
        rl_config["algorithm"]
    )
    # Load the entire eval_checkpoint configuration as a dictionary
    eval_checkpoint_config = config.get('eval_checkpoint', {})
    # Hyperparameters
    hyperparameters = rl_config["hyperparameters"]
    train = rl_config["train"]
    
    # data rollouts
    if train: 
        num_envs = rl_config["num_envs"]
        total_timesteps = rl_config["total_timesteps"]
        if num_envs > 1:
            logger.info("Running in RL mode with {} parallel environments.".format(num_envs))
            train_env = SubprocVecEnv([make_envs(simulation_config, i) for i in range(num_envs)])
        else:
            train_env = make_env(simulation_config)
        train_env.reset()
        model = algorithm_class(rl_config["policy_network"], \
                                train_env, \
                                **hyperparameters, \
                                policy_kwargs=policy_kwargs)
        # RL training mode
        # Create the custom checkpoint and evaluation callback
        eval_checkpoint_callback = EvalCheckpointCallback(exp_name=args.exp, **eval_checkpoint_config)
        # Train the model
        model.learn(total_timesteps=total_timesteps, callback=eval_checkpoint_callback\
                    , tb_log_name=args.exp)
        # Save the model
        model.save(eval_checkpoint_config["name_prefix"])
        return
    
    else:
        assert os.path.isfile(rl_config["episode_data"])
        logger.info("Testing the trained model on episode data {}".format(rl_config["episode_data"]))
        assert "eval_actions" in rl_config
        logger.info("Evaluating the model with actions/id: {}".format(rl_config["eval_actions"]))
        # RL testing mode
        with open(rl_config["episode_data"], "rb") as f:
            episode_data = pkl.load(f)
        logger.info("Loaded episode data with {} episodes.".format(len(episode_data.keys())))
        # Checkpoint evaluation
        rew_list = []
        success = []
        decision_step = {}
        succ_decision = {}
        for action, id in rl_config["eval_actions"].items():
            decision_step[id] = 0
            succ_decision[id] = 0
        vis_id = [] if "vis_id" not in rl_config else rl_config["vis_id"]
        worlds = {ts: None for ts in vis_id}
        # over write the checkpoint path if not none
        if args.checkpoint_path is not None:
            rl_config["checkpoint_path"] = args.checkpoint_path

        for ts in list(episode_data.keys()): 
            if (ts not in vis_id) and len(vis_id) > 0:
                continue
            logger.info("Evaluating episode {}...".format(ts))
            episode_cache = episode_data[ts]
            max_steps = 10000
            if "label_info" in episode_cache:
                logger.info("Episode label: {}".format(episode_cache["label_info"]))
            if not args.save_steps:
                assert "oracle_step" in episode_cache["label_info"], "Need oracle step for evaluation."
                max_steps = episode_cache["label_info"]["oracle_step"] * 2
            eval_env, cached_observation = make_env(simulation_config, episode_cache, True)
            if rl_config["algorithm"] == "ExpertCollector" or rl_config["algorithm"] == "Random":
                # expert and random agent do not need a policy network
                model = algorithm_class(eval_env)
            elif rl_config["algorithm"] in ["HRI", "NLM"]:
                # HRI and NLM are trained w/ ext code, just load the network
                model = algorithm_class(rl_config["policy_network"], \
                                        eval_env, \
                                        **hyperparameters, \
                                        policy_kwargs=policy_kwargs)
                model.load(rl_config["checkpoint_path"])
            else:
                # SB3-based agents
                policy_kwargs_use = copy.deepcopy(policy_kwargs)
                if rl_config["algorithm"] == 'A2C':
                    model = algorithm_class.load(rl_config["checkpoint_path"], \
                                    eval_env, **hyperparameters)
                else:
                    model = algorithm_class.load(rl_config["checkpoint_path"], \
                                    eval_env, **hyperparameters, policy_kwargs=policy_kwargs_use)
            logger.info("Loaded model from {}".format(rl_config["checkpoint_path"]))
            o = eval_env.init()
            rew = 0    
            step = 0   
            local_decision_step = {}
            local_succ_decision = {}
            for acc, id in rl_config["eval_actions"].items():
                local_decision_step[id] = 0
                local_succ_decision[id] = 1
            d = False
            while (not d) and (step < max_steps):
                step += 1
                oracle_action = eval_env.expert_action
                action, _ = model.predict(o, deterministic=True)
                # save step_wise decision succ per trajectory
                if oracle_action in local_decision_step.keys():
                    local_decision_step[oracle_action] = 1
                    if int(action) != oracle_action:
                        local_succ_decision[oracle_action] = 0
                o, r, d, i = eval_env.step(int(action))
                if ts in vis_id:
                    cached_observation["Time_Obs"][step] = i
                if i["Fail"][0]:
                    rew += r
                    break
                rew += r
            if i["success"]:
                success.append(1)
            else:
                success.append(0)
            for acc, id in rl_config["eval_actions"].items():
                if local_decision_step[id] == 0:
                    local_succ_decision[id] = 0
                decision_step[id] += local_decision_step[id]
                succ_decision[id] += local_succ_decision[id]
            if step >= max_steps:
                rew -= 3
            rew_list.append(rew)
            if args.save_steps:
                episode_cache["label_info"]['oracle_step'] = step
            logger.info("Episode {} took {} steps.".format(ts, step))
            logger.info("Episode {} achieved a score of {}".format(ts, rew))
            logger.info("Episode {} Success: {}".format(ts, success[-1]))
            logger.info("Episode {} Decision Step: {}".format(ts, local_decision_step))
            logger.info("Episode {} Success Decision: {}".format(ts, local_succ_decision))
            if ts in worlds.keys():
                worlds[ts] = cached_observation
        mean_reward = np.mean(rew_list)
        sr = np.mean(success)
        mSuccD, aSuccD, SuccDAct = cal_step_metric(decision_step, succ_decision)
        logger.info("Mean Score achieved: {}".format(mean_reward))
        logger.info("Success Rate: {}".format(sr))
        logger.info("Mean Decision Succ: {}".format(mSuccD))
        logger.info("Average Decision Succ: {}".format(aSuccD))
        logger.info("Decision Succ for each action: {}".format(SuccDAct))
        if args.save_steps:
            with open(os.path.join(args.log_dir, "{}_steps.pkl".format(args.exp)), "wb") as f:
                pkl.dump(episode_data, f)
        for ts in worlds.keys():
            if worlds[ts] is not None:
                with open(os.path.join(args.log_dir, "{}_{}.pkl".format(args.exp, ts)), "wb") as f:
                    pkl.dump(worlds[ts], f)

def cal_step_metric(decision_step, succ_decision):
    mean_decision_succ = {}
    total_decision = sum(decision_step.values())
    total_decision = max(total_decision, 1)
    total_succ = sum(succ_decision.values())
    for action, num in decision_step.items():
        num = max(num, 1)
        mean_decision_succ[action] = succ_decision[action]/num
    average_decision_succ = sum(mean_decision_succ.values())/len(mean_decision_succ)
    # mean decision succ (over all steps), average decision succ (over all actions), decision succ for each action
    return total_succ/total_decision, average_decision_succ, mean_decision_succ

def random_agents_generation(agent_num, priority_list, valid_concept_names, mode, stage):
    agents_list = []
    if mode == "easy":
        for agent_id in range(agent_num):
            agent = {}
            agent["id"] = agent_id
            # 3 for normal cars, 2 for normal pedestrians
            valid_concept_cnt = len(valid_concept_names)
            choice_range = valid_concept_cnt + 3 + 2
            concept_id = np.random.randint(choice_range)
            if concept_id < valid_concept_cnt:
                concept_name = valid_concept_names[concept_id]
                agent["class"] = STATIC_UNARY_PREDICATE_NAME_DICT[concept_name]["class"]
                agent["size"] = STATIC_UNARY_PREDICATE_NAME_DICT[concept_name]["size"]
                agent["gplanner"] = STATIC_UNARY_PREDICATE_NAME_DICT[concept_name]["gplanner"]
                agent["concepts"] = {}
                agent["concepts"]["type"] = STATIC_UNARY_PREDICATE_NAME_DICT[concept_name]["type"]
                agent["concepts"][STATIC_UNARY_PREDICATE_NAME_DICT[concept_name]["concept_name"]] = 1.0
            elif concept_id >= valid_concept_cnt and concept_id < valid_concept_cnt + 3:
                agent["class"] = "Private_car"
                agent["size"] = 2
                agent["gplanner"] = "A*vg"
                agent["concepts"] = {}
                agent["concepts"]["type"] = "Car"
            else:
                agent["class"] = "Pedestrian"
                agent["size"] = 1
                agent["gplanner"] = "A*"
                agent["concepts"] = {}
                agent["concepts"]["type"] = "Pedestrian"
            if agent["concepts"]["type"] == "Pedestrian":
                agent["concepts"]["priority"] = 0
            else:
                agent["concepts"]["priority"] = int(priority_list[0])
                priority_list = np.delete(priority_list, 0)
            agents_list.append(agent)

    elif mode == "expert":
        if stage != "test":
            ped_num = np.random.randint(4, 7)
        else:
            ped_num = np.random.randint(8, 11)
        # make sure we have 1 old, 2 young
        agent_old_1 = {
            "id": 0, "class": "Pedestrian", "size": 1, "gplanner": "A*",
            "concepts": {"type": "Pedestrian", "priority": 0, "old": 1.0},
        }
        agent_young_1 = {
            "id": 1, "class": "Pedestrian", "size": 1, "gplanner": "A*",
            "concepts": {"type": "Pedestrian", "priority": 0, "young": 1.0},
        }
        agent_young_2 = {
            "id": 2, "class": "Pedestrian", "size": 1, "gplanner": "A*",
            "concepts": {"type": "Pedestrian", "priority": 0, "young": 1.0},
        }
        agents_list.append(agent_old_1)
        agents_list.append(agent_young_1)
        agents_list.append(agent_young_2)
        # create random pedestrians, p(old):p(young):p(normal)=2:2:1
        for i in range(ped_num-3):
            agent = {
                "id": 3+i, "class": "Pedestrian", "size": 1, "gplanner": "A*",
                "concepts": {"type": "Pedestrian", "priority": 0},
            }
            tmp_id = np.random.randint(5)
            if tmp_id in [0, 1]:
                agent["concepts"]["old"] = 1.0
            elif tmp_id in [2, 3]:
                agent["concepts"]["young"] = 1.0
            agents_list.append(agent)
        # create random cars
        concepts_dist = {
            "concepts": ['ambulance', 'bus', 'police', 'reckless', 'tiro', 'normal'],
            "prob": [0.18, 0.18, 0.18, 0.18, 0.18, 0.1],
        }
        car_num = agent_num - ped_num
        for i in range(car_num):
            agent = {
                "id": ped_num+i, "class": "Private_car", "size": 2, "gplanner": "A*vg",
                "concepts": {"type": "Car", "priority": int(priority_list[0])},
            }
            priority_list = np.delete(priority_list, 0)
            import random
            sample = random.choices(concepts_dist['concepts'], weights=concepts_dist['prob'], k=1)[0]
            if sample != "normal":
                agent["concepts"][sample] = 1.0
            agents_list.append(agent)

    return agents_list

def create_vis_dataset(args, logger):
    """
    The format of this dataset pkl looks like:
        vis_dataset: {
            'World0_step0001': {
                'Image_path': 'vis_dataset/easy_1k/easy_1k_0_imgs/step_0001.png',
                'Predicate_groundings': {"IsAtInter": [0, 1, 0, ...], "HigherPri": [[0, 1, 0, ...],...], ...},
                'Bboxes': {0: (280, 424, 287, 431), 1: (576, 680, 583, 687), ...},
                'Types': {0: 'Car', 1: 'Pedestrian', ...},
                'Detailed_types': {0: 'normal_car', 1: 'old', ...},
                'Priorities': {0: 1, 1: 3, ...},
                'Directions': {0: 'left', 1: 'right', ...},
                'Next_actions': {0: 2, 1: 1, ...}, (0: Slow, 1: Normal, 2: Fast, 3: Stop)
            },
            ...
        }
    """
    if not os.path.exists(args.dataset_dir):
        os.makedirs(args.dataset_dir)
    world_num_dict = {
        "train": args.train_world_num,
        "val": args.val_world_num,
        "test": args.test_world_num,
    }
    print(args.config)
    config = load_config(args.config)
    tmp_agent_yaml_file = "{}/tmp_agents.yaml".format(args.dataset_dir)
    simulation_config = config["simulation"]
    print(simulation_config)

    # prepare agent concepts to choose from
    ontology_yaml_file = simulation_config["ontology_yaml_file"]
    with open(ontology_yaml_file, 'r') as file:
        ontology_config = yaml.load(file, Loader=yaml.Loader)
    valid_concept_names = []
    for predicate in ontology_config["Predicates"]:
        predicate_name = list(predicate.keys())[0]
        if predicate_name in STATIC_UNARY_PREDICATE_NAME_DICT:
            valid_concept_names.append(predicate_name)



    for stage, world_num in world_num_dict.items():
        if os.path.exists(os.path.join(args.dataset_dir, stage)):
            print("Dataset for {} already exists, skipping...".format(stage))
            continue

        # prepare icon img dir dict
        icon_dir_dict = {}
        for key in ICON_DIR_PATH_DICT.keys():
            if stage == "test":
                icon_dir_path = ICON_DIR_PATH_DICT[key]["test"]
            else:
                icon_dir_path = ICON_DIR_PATH_DICT[key]["train"]
            icon_num = len(os.listdir(icon_dir_path))
            icon_dir_dict[key] = {"icon_dir_path": icon_dir_path, "icon_num": icon_num}

        vis_dataset = {}
        if "fixed" in args.exp:
            simulation_config["agent_yaml_file"] = "config/agents/Vis/{}/{}.yaml".format(args.mode, stage)
            print("Using fixed expert agents from {}".format(args.exp))
        else:
            simulation_config["agent_yaml_file"] = tmp_agent_yaml_file
            print("Using random agents.")    
        # set min/max agent_num
        if stage == "train":
            simulation_config["agent_region"] = 100
            min_agent_num = args.min_agent_num_train
            max_agent_num = args.max_agent_num_train
        elif stage == "val": 
            simulation_config["agent_region"] = 100
            min_agent_num = args.min_agent_num_val
            max_agent_num = args.max_agent_num_val
        else:
            simulation_config["agent_region"] = 100
            min_agent_num = args.min_agent_num_test
            max_agent_num = args.max_agent_num_test
        # Stimulate several worlds
        for world_idx in trange(world_num):
            # set a suitable seed
            if stage == "train":
                seed = world_idx
            elif stage == "val": 
                seed = world_num_dict["train"] + world_idx
            else:
                seed = world_num_dict["train"] + world_num_dict["val"] + world_idx
            torch.manual_seed(seed)
            np.random.seed(seed)

            if "fixed" in args.exp:
                pass
            else:
                # Generate random agents
                agent_num = np.random.randint(min_agent_num, max_agent_num+1)
                priority_list = np.arange(1, agent_num+1)
                np.random.shuffle(priority_list)
                agents_list = random_agents_generation(agent_num, priority_list, valid_concept_names, args.mode, stage)
                with open(tmp_agent_yaml_file, "w") as f:
                    yaml.dump({"agents": agents_list}, f)

            # Main simulation loop
            city, cached_observation = CityLoader.from_yaml(**simulation_config)
            if "fixed" in args.exp:
                agent_num = len(cached_observation["Static Info"]["Agents"].keys())
            steps = 0
            while steps < args.max_steps:
                logger.info("Simulating Step_{}...".format(steps))
                s = time.time()
                time_obs = city.update()
                e = time.time()
                logger.info("Time spent: {}".format(e-s))
                steps += 1
                cached_observation["Time_Obs"][steps] = time_obs

            # Render imgs of current stimulated world and Repack the pkl of current stimulated world
            # crop_size = 1024 * simulation_config["agent_region"] // 100
            crop_size = 1024
            vis_dataset = pkl2city_imgs(cached_observation, vis_dataset, world_idx, icon_dir_dict, 
                output_folder = os.path.join(args.dataset_dir, "{}/world{}_agent{}_imgs".format(stage, world_idx, agent_num)),
                crop=[0, crop_size, 0, crop_size]
            )
        
        for step_name in list(vis_dataset.keys()):
            agent_out = False
            for bbox in list(vis_dataset[step_name]["Bboxes"].values()):
                if bbox[2] > crop_size or bbox[3] > crop_size:
                    agent_out = True
                    print(f"In stage {stage} step {step_name}, bbox {bbox} is out of the map region {crop_size}!")
                    break
            if agent_out:
                del vis_dataset[step_name]


        if not os.path.exists(os.path.join(args.dataset_dir, stage)):
            os.makedirs(os.path.join(args.dataset_dir, stage))
        with open(os.path.join(args.dataset_dir, "{}/{}_{}.pkl".format(stage, stage, args.exp)), "wb") as f:
            pkl.dump(vis_dataset, f)


if __name__ == '__main__':
    args = parse_arguments()
    logger = setup_logger(log_dir=args.log_dir, log_name=args.exp)
    if args.collect_only:
        logger.info("Running in data collection mode.")
        logger.info("Loading simulation config from {}.".format(args.config))
        main_collect(args, logger)
    elif args.use_gym:
        logger.info("Running in RL mode.")
        logger.info("Loading RL config from {}.".format(args.config))
        # RL mode, will use gym wrapper to learn and test an agent
        main_gym(args, logger)
    elif args.create_vis_dataset:
        logger.info("Running in vis dataset generation mode.")
        create_vis_dataset(args, logger)
    else:
        # Sim mode, will use the logic-based simulator to run a simulation (no learning)
        logger.info("Running in simulation mode.")
        logger.info("Loading simulation config from {}.".format(args.config))
        e = time.time()
        main(args, logger)
        logger.info("Total time spent: {}".format(time.time()-e))