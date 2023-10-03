import torch
from math import *
import networkx as nx
from utils.find import find_midroad_segments
from core.config import *

def manhattan_distance(u, v):
    return sqrt(pow(u[0] - v[0], 2) + pow(u[1] - v[1], 2))

def g_value(u, v):
    dis = torch.dist(u, v)
    if torch.all((u - v) != 0):
        dis += 30
    elif dis == 4:
        dis += 30
    return dis

class ASTAR_G:
    def __init__(self, movable_map, midline_matrix, offset):
        self.road_offset = offset
        self.movable_map = movable_map
        self.midroad_segments = find_midroad_segments(midline_matrix)
        self.build_graph(self.midroad_segments)

    def build_graph(self, road_segments):
        self.G = nx.DiGraph()  # Use a directed graph for one-directional edges
        # Connect each start point to its end point
        self.start_lists = []
        self.end_lists = []
        for segment in road_segments:
            mid_start, mid_end = segment
            if mid_start[0] == mid_end[0]:
                assert mid_end[1] > mid_start[1]
                # horizonal mid line
                bottom_s = mid_start + torch.tensor([self.road_offset, -1])
                bottom_e = mid_end + torch.tensor([self.road_offset, 1])
                self.G.add_edge(tuple(bottom_s.tolist()), tuple(bottom_e.tolist()), weight=g_value(bottom_s.float(), bottom_e.float()).item())
                self.start_lists.append(bottom_s.unsqueeze(0))
                self.end_lists.append(bottom_e.unsqueeze(0))
                top_s = mid_end + torch.tensor([-self.road_offset, 1])
                top_e = mid_start + torch.tensor([-self.road_offset, -1])
                self.G.add_edge(tuple(top_s.tolist()), tuple(top_e.tolist()), weight=g_value(top_s.float(), top_e.float()).item())
                self.start_lists.append(top_s.unsqueeze(0))
                self.end_lists.append(top_e.unsqueeze(0))
            elif mid_start[1] == mid_end[1]:
                assert mid_end[0] > mid_start[0]
                # vertical mid line
                left_s = mid_start + torch.tensor([-1, -self.road_offset])
                left_e = mid_end + torch.tensor([1, -self.road_offset])
                self.G.add_edge(tuple(left_s.tolist()), tuple(left_e.tolist()), weight=g_value(left_s.float(), left_e.float()).item())
                self.start_lists.append(left_s.unsqueeze(0))
                self.end_lists.append(left_e.unsqueeze(0))
                right_s = mid_end + torch.tensor([1, self.road_offset])
                right_e = mid_start + torch.tensor([-1, self.road_offset])
                self.G.add_edge(tuple(right_s.tolist()), tuple(right_e.tolist()), weight=g_value(right_s.float(), right_e.float()).item())
                self.start_lists.append(right_s.unsqueeze(0))
                self.end_lists.append(right_e.unsqueeze(0))
        
        # Connect each end point to other start points (this only happens in intersections)
        for end_point in self.end_lists:
            distances = torch.norm(torch.cat(self.start_lists, dim=0).float() - end_point.float(), dim=1)
            near_starts = torch.cat(self.start_lists, dim=0)[distances < 20]
            for starts in near_starts.tolist():
                self.G.add_edge(tuple(end_point.tolist()[0]), tuple(starts), weight=g_value(end_point.float(), torch.tensor(starts).float()).item())

    def find_nearest_node(self, point, origin_list = 's'):
        intersection = torch.zeros_like(point)
        next_node_list = self.start_lists if origin_list=='s' else self.end_lists
        judging_list = self.end_lists if origin_list=='s' else self.start_lists
        next_node_list = torch.cat(next_node_list, dim=0)
        judging_list = torch.cat(judging_list, dim=0)
        # which streets?
        all_nodes = torch.cat([next_node_list, judging_list], dim=0)
        dis = torch.norm(point.float() - all_nodes.float(), dim=1)
        min_dis_value, ind = torch.min(dis, dim=0)
        # min value lie in next node list, yes it is
        if torch.any((next_node_list[:, 0] == all_nodes[ind][0]) & (next_node_list[:, 1] == all_nodes[ind][1])):
            _, local_ind = torch.min(torch.abs(all_nodes[ind]-point), dim=0)
            intersection[1-local_ind] = point[1-local_ind]
            intersection[local_ind] = all_nodes[ind][local_ind]
            return intersection, all_nodes[ind]
        else:
            # min value doesn't lie in next node list
            next_node = torch.zeros_like(intersection)
            assert torch.any((judging_list[:, 0] == all_nodes[ind][0]) & (judging_list[:, 1] == all_nodes[ind][1]))
            if torch.abs(all_nodes[ind]-point)[0] == torch.abs(all_nodes[ind]-point)[1]:
                mo = self.movable_map[point[0], point[1]-2:point[1]+2]
                if torch.all(mo):
                    # horizonal
                    local_ind = 0
                else:
                    local_ind = 1
            else:
                _, local_ind = torch.min(torch.abs(all_nodes[ind]-point), dim=0)
            intersection[1-local_ind] = point[1-local_ind]
            intersection[local_ind] = all_nodes[ind][local_ind]

            next_node[local_ind] = all_nodes[ind][local_ind]
            filtered_node_list = next_node_list[next_node_list[:, local_ind]==all_nodes[ind][local_ind]]
            # find the goal that so not have start in between
            for k in range(filtered_node_list.shape[0]):
                candidate = filtered_node_list[k]
                if torch.abs(candidate[1-local_ind] - all_nodes[ind][1-local_ind]) == TRAFFIC_STREET_LENGTH + 1:
                    assert min(candidate[1-local_ind], all_nodes[ind][1-local_ind])<=intersection[1-local_ind]<=max(candidate[1-local_ind], all_nodes[ind][1-local_ind])
                    return intersection, candidate
            ValueError('Not a valid intersection')


    def plan(self, start, end):

        # Find intersections of start and end with their nearest road segments.
        intersect, close_goal = self.find_nearest_node(start, origin_list = 'g')
        self.G.add_edge(tuple(start.tolist()), tuple(intersect.tolist()))
        self.G.add_edge(tuple(intersect.tolist()), tuple(close_goal.tolist()))
        intersect, close_start = self.find_nearest_node(end)
        self.G.add_edge(tuple(close_start.tolist()), tuple(intersect.tolist()))
        self.G.add_edge(tuple(intersect.tolist()), tuple(end.tolist()))

        path_on_graph = nx.shortest_path(self.G, tuple(start.tolist()), tuple(end.tolist()), method='dijkstra')
        interpolated = self.interpolate(path_on_graph)
        # check
        for i in interpolated:
            assert self.movable_map[i[0], i[1]]

        return interpolated

    def interpolate(self, path_on_graph):
        interpolated = []

        for i in range(len(path_on_graph) - 1):
            current_point = path_on_graph[i]
            next_point = path_on_graph[i+1]

            while current_point != next_point:

                # Determine the difference in X and Y
                dx = next_point[0] - current_point[0]
                dy = next_point[1] - current_point[1]

                if dx != 0 and dy != 0:
                    # Move to the intersection if both dx and dy are non-zero
                    assert abs(dx) == abs(dy)
                    if self.movable_map[current_point[0], next_point[1]]:
                        intersect = (current_point[0], next_point[1])
                    else:
                        assert self.movable_map[next_point[0], current_point[1]]
                        intersect = (next_point[0], current_point[1])
                    # First move to intersect:
                    while current_point != intersect:
                        dx = intersect[0] - current_point[0]
                        dy = intersect[1] - current_point[1]
                        if abs(dx) >= 2:
                            step = 2 * int(dx/abs(dx))
                            current_point = (current_point[0] + step, current_point[1])
                        elif abs(dy) >= 2:
                            step = 2 * int(dy/abs(dy))
                            current_point = (current_point[0], current_point[1] + step)
                        else:
                            if dx != 0:
                                step = int(dx)
                                current_point = (current_point[0] + step, current_point[1])
                            elif dy != 0:
                                step = int(dy)
                                current_point = (current_point[0], current_point[1] + step)
                        interpolated.append(torch.tensor(current_point))
                else:
                    # If the vehicle doesn't need to turn, just move straight
                    if abs(dx) >= 2:
                        step = 2 * int(dx/abs(dx))
                        current_point = (current_point[0] + step, current_point[1])
                    elif abs(dy) >= 2:
                        step = 2 * int(dy/abs(dy))
                        current_point = (current_point[0], current_point[1] + step)
                    else:
                        if dx != 0:
                            step = int(dx)
                            current_point = (current_point[0] + step, current_point[1])
                        elif dy != 0:
                            step = int(dy)
                            current_point = (current_point[0], current_point[1] + step)
                    interpolated.append(torch.tensor(current_point))

        interpolated.append(torch.tensor(path_on_graph[-1]))  # Add the last point

        return interpolated