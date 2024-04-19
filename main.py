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
# from logicity.rl_agent.alg import *
# from logicity.utils.gym_wrapper import GymCityWrapper
# from stable_baselines3.common.vec_env import SubprocVecEnv
# from logicity.utils.gym_callback import EvalCheckpointCallback

def parse_arguments():
    parser = argparse.ArgumentParser(description='Logic-based city simulation.')
    # logger
    parser.add_argument('--log_dir', type=str, default="./log_sim")
    parser.add_argument('--exp', type=str, default="sim_easy2")
    parser.add_argument('--vis', action='store_true', help='Visualize the city.')
    # seed
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--max-steps', type=int, default=200)
    # RL
    parser.add_argument('--collect_only', action='store_true', help='Only collect expert data.')
    parser.add_argument('--use_gym', action='store_true', help='In gym mode, we can use RL alg. to control certain agents.')
    parser.add_argument('--save_steps', action='store_true', help='Save step-wise decision for each trajectory.')
    parser.add_argument('--config', default='config/tasks/sim/easy.yaml', help='Configure file for this RL exp.')
    parser.add_argument('--checkpoint_path', default=None, help='Path to the trained model.')
    # vis_dataset
    parser.add_argument('--create_vis_dataset', action='store_true', help='Create vis dataset from sim pkl.')
    parser.add_argument('--pkl_num', type=int, default=5)
    parser.add_argument('--dataset_dir', type=str, default="./vis_dataset/easy_1k")
    parser.add_argument('--img_dir', type=str, default="./vis_dataset/easy_1k")

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

    # prepare icon img dict
    icon_dict = {}
    for key in PATH_DICT.keys():
        if isinstance(PATH_DICT[key], list):
            raw_img = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in PATH_DICT[key]]
            resized_img = [resize_with_aspect_ratio(img, ICON_SIZE_DICT[key]) for img in raw_img]
            icon_dict[key] = resized_img
        else:
            raw_img = cv2.cvtColor(cv2.imread(PATH_DICT[key]), cv2.COLOR_BGR2RGB)
            resized_img = resize_with_aspect_ratio(raw_img, ICON_SIZE_DICT[key])
            icon_dict[key] = resized_img

    vis_dataset = dict()
    for world_idx in trange(args.pkl_num):
        pkl_path = os.path.join(args.log_dir, "{}_{}.pkl".format(args.exp, world_idx))
        with open(pkl_path, "rb") as f:
            cached_observation = pkl.load(f)
        # logger.info(cached_observation)
        last_icons = None
        for step in trange(1, args.max_steps):
            step_name = "World{}_step{:0>4d}".format(world_idx, step)
            vis_dataset[step_name] = {}
            vis_dataset[step_name]["Image_path"] = os.path.join(args.img_dir, "{}_{}_imgs".format(args.exp, world_idx) ,"step_{:0>4d}.png".format(step))
            vis_dataset[step_name]["Predicate_groundings"] = cached_observation["Time_Obs"][step]["Predicate_groundings"]
            vis_dataset[step_name]["Bboxes"] = {}
            vis_dataset[step_name]["Types"] = {}
            vis_dataset[step_name]["Detailed_types"] = {}
            vis_dataset[step_name]["Priorities"] = {}
            vis_dataset[step_name]["Directions"] = {}
            vis_dataset[step_name]["Next_actions"] = {}
            time_obs_world = cached_observation["Time_Obs"][step]["World"].numpy()
            time_obs_world_ = cached_observation["Time_Obs"][step+1]["World"].numpy()
            static_info_agents = list(cached_observation["Static Info"]["Agents"].values())
            agent_layer = time_obs_world[BASIC_LAYER:]
            agent_layer_ = time_obs_world_[BASIC_LAYER:]
            resized_grid = np.repeat(np.repeat(agent_layer, SCALE, axis=1), SCALE, axis=2)
            resized_grid_ = np.repeat(np.repeat(agent_layer_, SCALE, axis=1), SCALE, axis=2)
            # use icon_dict_local for the first time when last_icons is None
            icon_dict_local = {"icon": {}, "pos": {}}
            for agent_idx in range(resized_grid.shape[0]):
                # Get type and priority of agent 
                agent_type = static_info_agents[agent_idx]["concepts"]["type"]
                vis_dataset[step_name]["Types"][agent_idx] = agent_type
                vis_dataset[step_name]["Priorities"][agent_idx] = static_info_agents[agent_idx]["concepts"]["priority"]
                # Get detailed type of agent
                agent_detailed_type = list(set(list(DETAILED_TYPE_MAP.keys())) \
                                        & set(list(static_info_agents[agent_idx]["concepts"].keys())))
                if not agent_detailed_type:
                    if agent_type == "Car":
                        agent_detailed_type = "normal_car"
                    else:
                        agent_detailed_type = "normal_pedestrian"
                else:
                    agent_detailed_type = agent_detailed_type[0]
                vis_dataset[step_name]["Detailed_types"][agent_idx] = agent_detailed_type
                # Get bbox of agent icon (get direction by the way)
                # 1. get direction
                local_layer = resized_grid[agent_idx]
                local_layer_ = resized_grid_[agent_idx]
                left, top, right, bottom = get_pos(local_layer)
                left_, top_, right_, bottom_ = get_pos(local_layer_)
                direction = get_direction(left, left_, top, top_)
                vis_dataset[step_name]["Directions"][agent_idx] = direction
                # 2. get position
                pos = (left, top, right, bottom)
                # 3. get icon img array
                if agent_detailed_type.startswith('normal_'):
                    icon_list = icon_dict[agent_type].copy()
                    icon_id = agent_idx%len(icon_list)
                    icon = icon_list[icon_id]
                else:
                    icon = icon_dict[DETAILED_TYPE_MAP[agent_detailed_type]]
                if last_icons is None:
                    icon_img = Image.fromarray(icon)
                    icon_dict_local["icon"]["{}_{}".format(agent_type, agent_idx)] = [icon_img]
                    icon_bbox_left_last = None
                    icon_bbox_top_last = None
                else:
                    if direction == "none":
                        icon_img = last_icons["icon"]["{}_{}".format(agent_type, agent_idx)][1]
                        icon_bbox_left_last = last_icons["pos"]["{}_{}".format(agent_type, agent_idx)][0]
                        icon_bbox_top_last = last_icons["pos"]["{}_{}".format(agent_type, agent_idx)][1]
                    else:
                        icon_img = last_icons["icon"]["{}_{}".format(agent_type, agent_idx)][0]
                # 4. get street type and rotation settings
                if agent_type == "Car":
                    street_type = get_street_type(resized_grid, pos)
                    # define rotation angles for directions
                    rotation_angles = {'up': 0, 'right': 270, 'down': 180, 'left': 90, 'none': 0}
                else:
                    street_type = None
                    rotation_angles = {'up': 0, 'right': 0, 'down': -1, 'left': -1, 'none': 0}
                # 5. rotate the icon image based on the direction
                rotated_icon = rotate_image(icon_img, rotation_angles[direction])
                # 6. calc icon bbox
                if agent_type == "Car":
                    if direction == 'up':
                        icon_bbox_left = (left+right)//2 - rotated_icon.width//2
                        if street_type == "v" or street_type is None:
                            icon_bbox_top = top
                        else:
                            icon_bbox_top = bottom - rotated_icon.height
                    elif direction == 'right':
                        if street_type == "h" or street_type is None:
                            icon_bbox_left = right - rotated_icon.width
                        else:
                            icon_bbox_left = left
                        icon_bbox_top = (top+bottom)//2 - rotated_icon.height//2
                    elif direction == 'down':
                        icon_bbox_left = (left+right)//2 - rotated_icon.width//2
                        if street_type == "v" or street_type is None:
                            icon_bbox_top = bottom - rotated_icon.height
                        else:
                            icon_bbox_top = top
                    elif direction == 'left':
                        if street_type == "h" or street_type is None:
                            icon_bbox_left = left
                        else:
                            icon_bbox_left = right - rotated_icon.width
                        icon_bbox_top = (top+bottom)//2 - rotated_icon.height//2
                    elif direction == 'none':
                        if icon_bbox_left_last is not None:
                            icon_bbox_left = icon_bbox_left_last
                            icon_bbox_top = icon_bbox_top_last
                        else:
                            icon_bbox_left = left
                            icon_bbox_top = top
                elif agent_type == "Pedestrian":
                    if direction == "none":
                        if icon_bbox_left_last is not None:
                            icon_bbox_left = icon_bbox_left_last
                            icon_bbox_top = icon_bbox_top_last
                        else:
                            icon_bbox_left = left
                            icon_bbox_top = top
                    else:
                        icon_bbox_left = (left+right)//2 - rotated_icon.width//2
                        icon_bbox_top = (top+bottom)//2 - rotated_icon.height//2
                icon_bbox_right = icon_bbox_left + rotated_icon.width
                icon_bbox_bottom = icon_bbox_top + rotated_icon.height
                # 7. update last_icons
                if last_icons is None:
                    icon_dict_local["icon"]["{}_{}".format(agent_type, agent_idx)].append(rotated_icon)
                    icon_dict_local["pos"]["{}_{}".format(agent_type, agent_idx)] = (icon_bbox_left, icon_bbox_top)
                else:
                    last_icons["icon"]["{}_{}".format(agent_type, agent_idx)][1] = rotated_icon
                    last_icons["pos"]["{}_{}".format(agent_type, agent_idx)] = (icon_bbox_left, icon_bbox_top)

                vis_dataset[step_name]["Bboxes"][agent_idx] = (icon_bbox_left, icon_bbox_top, icon_bbox_right, icon_bbox_bottom)
                vis_dataset[step_name]["Next_actions"][agent_idx] = list(cached_observation["Time_Obs"][step]["Agent_actions"].values())[agent_idx]

            if last_icons is None:
                last_icons = icon_dict_local
 
        # render imgs of one stimulated world
        pkl2city_imgs(pkl_path, os.path.join(args.img_dir, "{}_{}_imgs".format(args.exp, world_idx)))

    if not os.path.exists(args.dataset_dir):
        os.makedirs(args.dataset_dir)
    with open(os.path.join(args.dataset_dir, "{}_{}.pkl".format(args.exp, args.pkl_num)), "wb") as f:
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