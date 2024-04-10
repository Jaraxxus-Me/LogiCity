from .basic import Agent
import torch
import numpy as np
import random
import torch.nn.functional as F
from torch.distributions import Categorical
from ..utils.gen import gen_occ
from ..utils.find import find_nearest_building, find_building_mask
from ..utils.sample import sample_start_goal, sample_start_goal_vh, sample_determine_start_goal
from ..planners import GPlanner_mapper
from ..core.config import *
import logging
# import cv2
# import numpy as np
# # vis quick tool
# vis = np.zeros((250, 250, 3))
# vis[self.movable_region] = [255, 0, 0]
# vis[self.goal[0], self.goal[1]] = [0, 255, 255]
# vis[self.start[0], self.start[1]] = [0, 255, 0]
# vis[self.midline_matrix] = [0, 0, 255]
# cv2.imwrite("test.png", vis)

logger = logging.getLogger(__name__)

TYPE_MAP = {v: k for k, v in LABEL_MAP.items()}

class Car(Agent):
    def __init__(self, size, id, world_state_matrix, global_planner, concepts, init_info=None, debug=False, region=240):
        self.start_point_list = None
        self.goal_point_list = None
        self.global_planner_type = global_planner
        super().__init__(size, id, world_state_matrix, concepts, init_info=init_info, debug=debug, region=region)
        # Actions: ["left_1", "right_1", "up_1", "down_1", "left_2", "right_2", "up_2", "down_2", "left_3", "right_3", "up_3", "down_3", "stop"]
        self.action_space = torch.tensor(range(13))
        self.action_to_move = {
            self.action_space[0].item(): torch.tensor((0, -1)),
            self.action_space[1].item(): torch.tensor((0, 1)),
            self.action_space[2].item(): torch.tensor((-1, 0)),
            self.action_space[3].item(): torch.tensor((1, 0)),
            self.action_space[4].item(): torch.tensor((0, -2)),
            self.action_space[5].item(): torch.tensor((0, 2)),
            self.action_space[6].item(): torch.tensor((-2, 0)),
            self.action_space[7].item(): torch.tensor((2, 0)),
            self.action_space[8].item(): torch.tensor((0, -3)),
            self.action_space[9].item(): torch.tensor((0, 3)),
            self.action_space[10].item(): torch.tensor((-3, 0)),
            self.action_space[11].item(): torch.tensor((3, 0))
        }
        self.move_to_action = {
            self.action_to_move[k]: k for k in self.action_to_move
        }
        self.action_dist = torch.zeros_like(self.action_space).float()
        self.action_mapping = {
            0: "Left_Slow", 
            1: "Right_Slow", 
            2: "Up_Slow", 
            3: "Down_Slow", 
            4: "Left_Normal", 
            5: "Right_Normal", 
            6: "Up_Normal", 
            7: "Down_Normal",
            8: "Left_Fast", 
            9: "Right_Fast", 
            10: "Up_Fast", 
            11: "Down_Fast",
            12: "Stop"
            }

    def init(self, world_state_matrix, init_info=None, debug=False):
        Traffic_STREET = TYPE_MAP['Traffic Street']
        CROSSING_STREET = TYPE_MAP['Overlap']
        # specify the occupacy map
        self.movable_region = (world_state_matrix[STREET_ID] == Traffic_STREET) | (world_state_matrix[STREET_ID] == CROSSING_STREET)
        self.midline_matrix = (world_state_matrix[STREET_ID] == Traffic_STREET+MID_LINE_CODE_PLUS)
        self.global_planner = GPlanner_mapper[self.global_planner_type](self.movable_region, self.midline_matrix, CAR_STREET_OFFSET)
        self.intersection_points = torch.cat([torch.cat(self.global_planner.start_lists, dim=0), torch.cat(self.global_planner.end_lists, dim=0)], dim=0)
        if init_info is not None:
            self.init_from_dict(init_info)
            _ = self.get_start(world_state_matrix)
            _ = self.get_goal(world_state_matrix, self.start)
        else:
            if debug:
                self.start, self.goal = sample_determine_start_goal(self.type, self.id)
                self.pos = self.start.clone()
            else:
                self.start = torch.tensor(self.get_start(world_state_matrix))
                self.pos = self.start.clone()
                self.goal = torch.tensor(self.get_goal(world_state_matrix, self.start))
        # get global traj on the occupacy map
        self.global_traj = self.global_planner.plan(self.start, self.goal, 1)
        self.reach_goal = False
        self.last_move_dir = None
        logger.info("{}_{} initialization done!".format(self.type, self.id))

    def reset_concepts(self, max_priority, concepts_dist=None):
        self.priority = np.random.randint(1, max_priority)
        if concepts_dist is not None:
            self.concepts = {
                "type": "Car"
            }
            sample = random.choices(concepts_dist['concepts'], weights=concepts_dist['prob'], k=1)[0]
            self.concepts["priority"] = self.priority
            if sample == "normal":
                return
            else:
                self.concepts[sample] = 1.0
                return
        else:
            self.concepts["priority"] = self.priority
            return

    def get_start(self, world_state_matrix):
        # Define the labels for different entities
        building = [TYPE_MAP[b] for b in CAR_GOAL_START]
        # Find cells that are walking streets and have a house or office around them
        desired_locations = sample_start_goal_vh(world_state_matrix, TYPE_MAP['Traffic Street'], building, kernel_size=CAR_GOAL_START_INCLUDE_KERNEL)
        desired_locations[self.region:, :] = False
        desired_locations[:, self.region:] = False
        desired_locations = desired_locations * self.global_planner.permitted_sg

        self.start_point_list = torch.nonzero(desired_locations).tolist()
        random_index = torch.randint(0, len(self.start_point_list), (1,)).item()
        
        # Fetch the corresponding location
        start_point = self.start_point_list[random_index]

        # Return the indices of the desired locations
        return start_point

    def get_goal(self, world_state_matrix, start_point):
        # Define the labels for different entities
        # Define the labels for different entities
        building = [TYPE_MAP[b] for b in CAR_GOAL_START]

        # Find cells that are walking streets and have a house, office, or store around them
        self.desired_locations = desired_locations = sample_start_goal_vh(world_state_matrix, TYPE_MAP['Traffic Street'], building, kernel_size=CAR_GOAL_START_INCLUDE_KERNEL)
        desired_locations = self.desired_locations.detach().clone()

        # Determine the nearest building to the start point
        nearest_building = find_nearest_building(world_state_matrix, start_point)
        start_block = world_state_matrix[BLOCK_ID][nearest_building[0], nearest_building[1]]

        # Get the mask for the building containing the nearest_building position
        # building_mask = find_building_mask(world_state_matrix, nearest_building)
        building_mask = world_state_matrix[BLOCK_ID] == start_block
        
        # Create a mask to exclude areas around the building. We'll dilate the building mask.
        exclusion_radius = CAR_GOAL_START_EXCLUDE_KERNEL  # Excludes surrounding 3 grids around the building
        expanded_mask = F.max_pool2d(building_mask[None, None].float(), exclusion_radius, stride=1, padding=(exclusion_radius - 1) // 2) > 0
        
        desired_locations[expanded_mask[0, 0]] = False
        desired_locations[self.region:, :] = False
        desired_locations[:, self.region:] = False
        desired_locations[start_point[0], start_point[1]] = False
        desired_locations = desired_locations * self.global_planner.permitted_sg

        # Return the indices of the desired locations
        goal_point_list = torch.nonzero(desired_locations).tolist()
        random_index = torch.randint(0, len(goal_point_list), (1,)).item()
        
        # Fetch the corresponding location
        goal_point = goal_point_list[random_index]
        assert (goal_point[0] != start_point[0] or goal_point[1] != start_point[1]), "start point and goal point are the same!"

        # Return the indices of the desired locations
        return goal_point

    def get_next_action(self, world_state_matrix, local_action_dist):
        # reached goal
        if not self.reach_goal:
            if torch.all(self.pos == self.goal):
                if not self.debug:
                    self.reach_goal = True
                    self.reach_goal_buffer += REACH_GOAL_WAITING
                    # logger.info("{}_{} reached goal! Will change goal in the next {} step!".format(self.type, self.id, self.reach_goal_buffer))
                # else:
                    # logger.info("{}_{} reached goal! In Debug, it will stop".format(self.type, self.id))
                return self.action_space[-1], world_state_matrix[self.layer_id]
            else:
                return self.get_action(local_action_dist), world_state_matrix[self.layer_id]
        else:
            if self.reach_goal_buffer > 0:
                self.reach_goal_buffer -= 1
                # logger.info("{}_{} reached goal! Will change goal in the next {} step!".format(self.type, self.id, self.reach_goal_buffer))
                return self.action_space[-1], world_state_matrix[self.layer_id]
            # logger.info("Generating new goal and gloabl plans for {}_{}...".format(self.type, self.id))
            self.start = self.goal.clone()
            desired_locations = self.desired_locations.detach().clone()

            # Determine the nearest building to the start point
            nearest_building = find_nearest_building(world_state_matrix, self.start)
            start_block = world_state_matrix[BLOCK_ID][nearest_building[0], nearest_building[1]]

            # Get the mask for the building containing the nearest_building position
            building_mask = world_state_matrix[BLOCK_ID] == start_block
            
            # Create a mask to exclude areas around the building. We'll dilate the building mask.
            exclusion_radius = CAR_GOAL_START_EXCLUDE_KERNEL  # Excludes surrounding 3 grids around the building
            expanded_mask = F.max_pool2d(building_mask[None, None].float(), exclusion_radius, stride=1, padding=(exclusion_radius - 1) // 2) > 0
            
            desired_locations[expanded_mask[0, 0]] = False
            desired_locations[self.region:, :] = False
            desired_locations[:, self.region:] = False

            # Return the indices of the desired locations
            goal_point_list = torch.nonzero(desired_locations).tolist()
            random_index = torch.randint(0, len(goal_point_list), (1,)).item()
            
            # Fetch the corresponding location
            self.goal = torch.tensor(goal_point_list[random_index])
            self.global_traj = self.global_planner.plan(self.start, self.goal, 1)
            # logger.info("Generating new goal and gloabl plans for {}_{} done!".format(self.type, self.id))
            self.reach_goal = False
            self.last_move_dir = None
            # delete past traj
            world_state_matrix[self.layer_id] *= 0
            world_state_matrix[self.layer_id][self.goal[0], self.goal[1]] = TYPE_MAP[self.type] + AGENT_GOAL_PLUS
            for way_points in self.global_traj[1:-1]:
                world_state_matrix[self.layer_id][way_points[0], way_points[1]] \
                    = TYPE_MAP[self.type] + AGENT_GLOBAL_PATH_PLUS
            world_state_matrix[self.layer_id][self.start[0], self.start[1]] = TYPE_MAP[self.type]
            return self.get_action(local_action_dist), world_state_matrix[self.layer_id]

    def get_global_action(self):
        global_action_dist = torch.zeros_like(self.action_space).float()
        current_pos = torch.all((self.global_traj == self.pos), dim=1).nonzero()[0]
        next_pos = current_pos + 1 if current_pos < len(self.global_traj) - 1 else 0
        del_pos = self.global_traj[next_pos] - self.pos
        for move in self.move_to_action.keys():
            if torch.dot(del_pos.squeeze(), move) > 0:
                next_point = self.pos + move
                step = torch.max(torch.abs(move)).item()
                # for cars, it can't bypass the intersection line, check this, reject the move if it go bypass the line
                if step > 1:
                    if self.move_bypass(self.pos, next_point):
                        continue
                # two metrics:
                # 1. if the next point is on the global traj
                if len(torch.all((self.global_traj[next_pos:next_pos+step] == next_point), dim=1).nonzero()) > 0:
                    global_action_dist[self.move_to_action[move]] = 1.0
        if torch.all(global_action_dist==0):
            global_action_dist[-1] = 1.0
            if torch.all(del_pos==0):
                self.global_traj = torch.cat([self.global_traj[:next_pos], self.global_traj[next_pos+1:]], dim=0)
        return global_action_dist

    def move_bypass(self, A, B):
        # Compute the vectors
        AB = B - A
        AP = self.intersection_points - A.unsqueeze(0)
        BP = self.intersection_points - B.unsqueeze(0)
        BA = -AB

        # Check if points are on the line defined by A and B
        area = AB[0] * (self.intersection_points[:, 1] - A[1]) - AB[1] * (self.intersection_points[:, 0] - A[0])
        on_line = (area == 0)

        # Check if points are on the segment AB (between A and B)
        on_segment = (torch.matmul(AP, AB.unsqueeze(1)).squeeze() > 0) & (torch.matmul(BP, BA.unsqueeze(1)).squeeze() > 0)

        # Final mask where both conditions are true
        on_line_segment = on_line & on_segment

        # Check if any point satisfies the condition
        has_point_on_segment = on_line_segment.any().item()
        return has_point_on_segment