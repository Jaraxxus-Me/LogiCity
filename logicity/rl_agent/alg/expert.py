import pickle
import copy

import logging
logger = logging.getLogger(__name__)

class ExpertCollector:
    def __init__(self, env, num_episodes=1, return_full_world=False):
        """
        Expert data collector for rollouts using an expert policy, collecting data by trajectory with dictionaries.
        
        :param env: The gym environment wrapped with GymCityWrapper.
        :param num_episodes: Total number of episodes to collect data for.
        """
        self.env = env
        self.num_episodes = num_episodes
        self.return_full_world = return_full_world
        if return_full_world:
            self.full_world = []
        self.trajectories = []

    def collect_data(self, cached_observation):
        """
        Execute rollouts in the environment using the expert policy and collect data by trajectory,
        where each step in a trajectory is saved as a dictionary.
        
        :return: Collected data as a list of trajectories, each trajectory is a list of dictionaries
                 with keys ['state', 'action', 'reward', 'next_state', 'done'].
        """
        total_steps = 0
        for episode in range(self.num_episodes):
            logger.info(f"Collecting data for episode {episode + 1}/{self.num_episodes}...")
            obs = self.env.reset()
            done = False
            trajectory = []
            cuurent_step = 0
            world = copy.deepcopy(cached_observation)
            while not done:
                total_steps += 1
                cuurent_step += 1
                action = self.env.expert_action  # Assuming this gives the expert action directly from the environment
                new_obs, reward, done, info = self.env.step(action)
                world["Time_Obs"][cuurent_step] = info
                
                # Store the step in the current trajectory as a dictionary
                trajectory.append({
                    'state': obs,
                    'action': action,
                    'reward': reward,
                    'next_state': new_obs,
                    'done': done
                })
                
                obs = new_obs
            # Store the complete trajectory
            logger.info(f"Trajectory {episode + 1} collected with {len(trajectory)} steps. Total steps: {total_steps}")
            self.trajectories.append(trajectory)
            if self.return_full_world:
                self.full_world.append(world)
        if self.return_full_world:
            return self.trajectories, self.full_world
        else:
            return self.trajectories, None

    def save_data(self, filename):
        """
        Save the collected trajectories to a file for later use, with each step represented as a dictionary.
        
        :param filename: The name of the file where to save the data.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.trajectories, f, protocol=pickle.HIGHEST_PROTOCOL)

    def predict(self, observation, deterministic=True):
        """
        Predict the expert action for the given observation.

        :param observation: The current observation from the environment.
        :param deterministic: A flag indicating whether the prediction should be deterministic. For an expert policy,
                              this can typically be ignored as expert actions are deterministic by nature.
        :return: The predicted action and the internal state which is None for the expert policy.
        """
        # Assuming the environment's expert policy does not depend on the observation
        # directly and is determined by the internal state of the environment.
        action = self.env.expert_action
        return action, None