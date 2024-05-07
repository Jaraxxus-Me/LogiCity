import os
import numpy as np
import time
import random
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle as pkl
from tqdm import trange
from scipy.ndimage import label
from ..core.config import *


STATIC_UNARY_PREDICATE_NAME_DICT = {
    "IsAmbulance": {
        "concept_name": "ambulance",
        "class": "Private_car",
        "type": "Car",
        "size": 2,
        "gplanner": "A*vg",
    },
    "IsBus": {
        "concept_name": "bus",
        "class": "Private_car",
        "type": "Car",
        "size": 2,
        "gplanner": "A*vg",
    },
    "IsPolice": {
        "concept_name": "police",
        "class": "Private_car",
        "type": "Car",
        "size": 2,
        "gplanner": "A*vg",
    },
    "IsTiro": {
        "concept_name": "tiro",
        "class": "Private_car",
        "type": "Car",
        "size": 2,
        "gplanner": "A*vg",
    },
    "IsReckless": {
        "concept_name": "reckless",
        "class": "Private_car",
        "type": "Car",
        "size": 2,
        "gplanner": "A*vg",
    },
    "IsOld": {
        "concept_name": "old",
        "class": "Pedestrian",
        "type": "Pedestrian",
        "size": 1,
        "gplanner": "A*",
    },
    "IsYoung": {
        "concept_name": "young",
        "class": "Pedestrian",
        "type": "Pedestrian",
        "size": 1,
        "gplanner": "A*",
    },
}

IMAGE_BASE_PATH = "./imgs"
# IMAGE_BASE_PATH = "./imgs_no_variance"

ICON_DIR_PATH_DICT = {
    "Car": {
        "train": os.path.join(IMAGE_BASE_PATH, "train", "car_normal"),
        "test": os.path.join(IMAGE_BASE_PATH, "test", "car_normal"),
    },
    "Ambulance": {
        "train": os.path.join(IMAGE_BASE_PATH, "train", "car_ambulance"),
        "test": os.path.join(IMAGE_BASE_PATH, "test", "car_ambulance"),
    },
    "Bus": {
        "train": os.path.join(IMAGE_BASE_PATH, "train", "car_bus"),
        "test": os.path.join(IMAGE_BASE_PATH, "test", "car_bus"),
    },
    "Tiro": {
        "train": os.path.join(IMAGE_BASE_PATH, "train", "car_tiro"),
        "test": os.path.join(IMAGE_BASE_PATH, "test", "car_tiro"),
    },
    "Police": {
        "train": os.path.join(IMAGE_BASE_PATH, "train", "car_police"),
        "test": os.path.join(IMAGE_BASE_PATH, "test", "car_police"),
    },
    "Reckless": {
        "train": os.path.join(IMAGE_BASE_PATH, "train", "car_reckless"),
        "test": os.path.join(IMAGE_BASE_PATH, "test", "car_reckless"),
    },
    "Pedestrian": {
        "train": os.path.join(IMAGE_BASE_PATH, "train", "pedestrian_normal"),
        "test": os.path.join(IMAGE_BASE_PATH, "test", "pedestrian_normal"),
    },
    "Pedestrian_old": {
        "train": os.path.join(IMAGE_BASE_PATH, "train", "pedestrian_old"),
        "test": os.path.join(IMAGE_BASE_PATH, "test", "pedestrian_old"),
    },
    "Pedestrian_young": {
        "train": os.path.join(IMAGE_BASE_PATH, "train", "pedestrian_young"),
        "test": os.path.join(IMAGE_BASE_PATH, "test", "pedestrian_young"),
    },
    "Walking Street": {
        "train": os.path.join(IMAGE_BASE_PATH, "train", "walking"),
        "test": os.path.join(IMAGE_BASE_PATH, "test", "walking"),
    },
    "Traffic Street": {
        "train": os.path.join(IMAGE_BASE_PATH, "train", "traffic"),
        "test": os.path.join(IMAGE_BASE_PATH, "test", "traffic"),
    },
    "Overlap": {
        "train": os.path.join(IMAGE_BASE_PATH, "train", "crossing"),
        "test": os.path.join(IMAGE_BASE_PATH, "test", "crossing"),
    },
    "Gas Station": {
        "train": os.path.join(IMAGE_BASE_PATH, "train", "gas"),
        "test": os.path.join(IMAGE_BASE_PATH, "test", "gas"),
    },
    "Garage": {
        "train": os.path.join(IMAGE_BASE_PATH, "train", "garage"),
        "test": os.path.join(IMAGE_BASE_PATH, "test", "garage"),
    },
    "House": {
        "train": os.path.join(IMAGE_BASE_PATH, "train", "house"),
        "test": os.path.join(IMAGE_BASE_PATH, "test", "house"),
    },
    "Office": {
        "train": os.path.join(IMAGE_BASE_PATH, "train", "office"),
        "test": os.path.join(IMAGE_BASE_PATH, "test", "office"),
    },
    "Store": {
        "train": os.path.join(IMAGE_BASE_PATH, "train", "store"),
        "test": os.path.join(IMAGE_BASE_PATH, "test", "store"),
    },
}

ICON_DIR_PATH_DICT_ALL = {
    "Car": {
        "train": os.path.join(IMAGE_BASE_PATH, "all", "car_normal"),
        "test": os.path.join(IMAGE_BASE_PATH, "all", "car_normal"),
    },
    "Ambulance": {
        "train": os.path.join(IMAGE_BASE_PATH, "all", "car_ambulance"),
        "test": os.path.join(IMAGE_BASE_PATH, "all", "car_ambulance"),
    },
    "Bus": {
        "train": os.path.join(IMAGE_BASE_PATH, "all", "car_bus"),
        "test": os.path.join(IMAGE_BASE_PATH, "all", "car_bus"),
    },
    "Tiro": {
        "train": os.path.join(IMAGE_BASE_PATH, "all", "car_tiro"),
        "test": os.path.join(IMAGE_BASE_PATH, "all", "car_tiro"),
    },
    "Police": {
        "train": os.path.join(IMAGE_BASE_PATH, "all", "car_police"),
        "test": os.path.join(IMAGE_BASE_PATH, "all", "car_police"),
    },
    "Reckless": {
        "train": os.path.join(IMAGE_BASE_PATH, "all", "car_reckless"),
        "test": os.path.join(IMAGE_BASE_PATH, "all", "car_reckless"),
    },
    "Pedestrian": {
        "train": os.path.join(IMAGE_BASE_PATH, "all", "pedestrian_normal"),
        "test": os.path.join(IMAGE_BASE_PATH, "all", "pedestrian_normal"),
    },
    "Pedestrian_old": {
        "train": os.path.join(IMAGE_BASE_PATH, "all", "pedestrian_old"),
        "test": os.path.join(IMAGE_BASE_PATH, "all", "pedestrian_old"),
    },
    "Pedestrian_young": {
        "train": os.path.join(IMAGE_BASE_PATH, "all", "pedestrian_young"),
        "test": os.path.join(IMAGE_BASE_PATH, "all", "pedestrian_young"),
    },
    "Walking Street": {
        "train": os.path.join(IMAGE_BASE_PATH, "all", "walking"),
        "test": os.path.join(IMAGE_BASE_PATH, "all", "walking"),
    },
    "Traffic Street": {
        "train": os.path.join(IMAGE_BASE_PATH, "all", "traffic"),
        "test": os.path.join(IMAGE_BASE_PATH, "all", "traffic"),
    },
    "Overlap": {
        "train": os.path.join(IMAGE_BASE_PATH, "all", "crossing"),
        "test": os.path.join(IMAGE_BASE_PATH, "all", "crossing"),
    },
    "Gas Station": {
        "train": os.path.join(IMAGE_BASE_PATH, "all", "gas"),
        "test": os.path.join(IMAGE_BASE_PATH, "all", "gas"),
    },
    "Garage": {
        "train": os.path.join(IMAGE_BASE_PATH, "all", "garage"),
        "test": os.path.join(IMAGE_BASE_PATH, "all", "garage"),
    },
    "House": {
        "train": os.path.join(IMAGE_BASE_PATH, "all", "house"),
        "test": os.path.join(IMAGE_BASE_PATH, "all", "house"),
    },
    "Office": {
        "train": os.path.join(IMAGE_BASE_PATH, "all", "office"),
        "test": os.path.join(IMAGE_BASE_PATH, "all", "office"),
    },
    "Store": {
        "train": os.path.join(IMAGE_BASE_PATH, "all", "store"),
        "test": os.path.join(IMAGE_BASE_PATH, "all", "store"),
    },
}

SCALE = 8

ICON_SIZE_DICT = {
    "Car": SCALE*6,
    "Ambulance": SCALE*6,
    "Bus": SCALE*6,
    "Tiro": SCALE*6,
    "Police": SCALE*6,
    "Reckless": SCALE*6,
    "Pedestrian": SCALE*4,
    "Pedestrian_old": SCALE*4,
    "Pedestrian_young": SCALE*4,
    "Walking Street": SCALE*10,
    "Traffic Street": SCALE*10,
    "Overlap": SCALE*10,
    "Gas Station": SCALE*BUILDING_SIZE,
    "Garage": SCALE*BUILDING_SIZE,
    "House": SCALE*BUILDING_SIZE,
    "Office": SCALE*BUILDING_SIZE,
    "Store": SCALE*BUILDING_SIZE,
}


def visualize_city(city, resolution, agent_layer=None, file_name="city.png"):
    # Define a color for each entity
    # only visualize a static map with buildings, streets, and one layer agents
    # Create a visual grid of the city, buildings and streets
    visual_grid = np.ones((resolution, resolution, 3), dtype=np.uint8)*200
    np_grid = city.city_grid.numpy().astype(np.float32)
    scale_factor = resolution/city.grid_size[0]
    for k in range(1, 3):
        for i in range(city.grid_size[0]):
            for j in range(city.grid_size[1]):
                if np_grid[k][i][j] != 0:
                    color = city.color_map[np_grid[k][i][j]]
                    visual_grid[int(i*scale_factor):int((i+1)*scale_factor), int(j*scale_factor):int((j+1)*scale_factor)] = color

    # draw agent's if provided layer
    if agent_layer != None:
        visual_grid = vis_agent(visual_grid, city, agent_layer, scale_factor)
    
    # Add the legend
    padding = int(5*scale_factor)
    column_width = int(70*scale_factor)
    legend_width = resolution
    legend_height = resolution
    legend_item_height = int(10*scale_factor)
    legend_img = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255  # white background

    for idx, (key, value) in enumerate(city.color_map.items()):
        c = idx//23
        idx = idx%23
        y_offset = idx * legend_item_height + padding
        if y_offset + legend_item_height > legend_height:  # Ensure we don't render beyond the legend image
            break
        cv2.rectangle(legend_img, (padding+column_width*c, y_offset), (padding+column_width*c + legend_item_height, y_offset + legend_item_height), city.color_map[key], -1)
        if key in city.label2type.keys():
            cv2.putText(legend_img, str(city.label2type[key]), (padding+column_width*c + int(15*scale_factor), y_offset + legend_item_height - int(3*scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, 0.2*scale_factor, \
                (0, 0, 0), 2, lineType=cv2.LINE_AA)
        else:
            cv2.putText(legend_img, key, (padding+column_width*c + int(15*scale_factor), y_offset + legend_item_height - int(3*scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, 0.2*scale_factor, \
                (0, 0, 0), 2, lineType=cv2.LINE_AA)

    # Combine the visual grid and the legend side by side
    combined_img = np.hstack((visual_grid, legend_img))
    # just visualize the left corner that is constructed
    # combined_img = combined_img[:int(resolution/2), :int(resolution/2), :]

    # Use OpenCV to display the city
    cv2.imwrite(file_name, combined_img)

def vis_agent(vis_grid, city, agent_layer_ids, scale_factor, curr=True, s=True, g=True, path=True):
    if agent_layer_ids == -1:
        agent_layer_ids = list(range(BASIC_LAYER, city.city_grid.shape[0]))
    if isinstance(agent_layer_ids, list):
        # visualize multi-ple agents
        for agent_layer_id in agent_layer_ids:
            agent_type = city.agents[agent_layer_id-BASIC_LAYER].type
            agent_id = city.agents[agent_layer_id-BASIC_LAYER].id
            agent_layer = city.city_grid[agent_layer_id]
            agent_color = city.color_map["{}_{}".format(agent_type, agent_id)]
            # get points
            cur_agent_pos = torch.nonzero((agent_layer == city.type2label[agent_type]).float()).tolist()
            planned_traj = torch.nonzero((agent_layer == city.type2label[agent_type]+AGENT_GLOBAL_PATH_PLUS).float()).tolist()
            walked_traj = torch.nonzero((agent_layer == city.type2label[agent_type]+AGENT_WALKED_PATH_PLUS).float()).tolist()
            goal_agent_pos = torch.nonzero((agent_layer == city.type2label[agent_type]+AGENT_GOAL_PLUS).float()).tolist()
            start_agent_pos = torch.nonzero((agent_layer == city.type2label[agent_type]+AGENT_START_PLUS).float()).tolist()
            # Path
            if len(planned_traj) > 0 and path:  
                for way_point in planned_traj:
                    vis_grid[int(way_point[0]*scale_factor):int((way_point[0]+1)*scale_factor), \
                        int(way_point[1]*scale_factor):int((way_point[1]+1)*scale_factor)] = agent_color
            if len(walked_traj) > 0 and path:  
                for way_point_ in walked_traj:
                    vis_grid[int(way_point_[0]*scale_factor):int((way_point_[0]+1)*scale_factor), \
                        int(way_point_[1]*scale_factor):int((way_point_[1]+1)*scale_factor)] = agent_color
            # Points
            if len(start_agent_pos) > 0 and s:
                cv2.drawMarker(vis_grid, (int(start_agent_pos[0][1]*scale_factor), int(start_agent_pos[0][0]*scale_factor)), \
                    agent_color, markerType=cv2.MARKER_TRIANGLE_UP, markerSize=14, thickness=3, line_type=cv2.LINE_AA)
            if len(goal_agent_pos) > 0 and g:   
                cv2.drawMarker(vis_grid, (int(goal_agent_pos[0][1]*scale_factor), int(goal_agent_pos[0][0]*scale_factor)), \
                    agent_color, markerType=cv2.MARKER_TRIANGLE_DOWN, markerSize=14, thickness=3, line_type=cv2.LINE_AA)
            assert len(cur_agent_pos) == 1
            if curr:
                cv2.drawMarker(vis_grid, (int(cur_agent_pos[0][1]*scale_factor), int(cur_agent_pos[0][0]*scale_factor)), \
                    agent_color, markerType=cv2.MARKER_DIAMOND, markerSize=14, thickness=3, line_type=cv2.LINE_AA)

    elif isinstance(agent_layer_ids, int):
        agent_layer_id = agent_layer_ids
        assert agent_layer_id >= BASIC_LAYER
        agent_type = city.agents[agent_layer_id-BASIC_LAYER].type
        agent_id = city.agents[agent_layer_id-BASIC_LAYER].id
        agent_layer = city.city_grid[agent_layer_id]
        agent_color = city.color_map["{}_{}".format(agent_type, agent_id)]
        # draw points
        cur_agent_pos = torch.nonzero((agent_layer == city.type2label[agent_type]).float()).tolist()
        planned_traj = torch.nonzero((agent_layer == city.type2label[agent_type]+AGENT_GLOBAL_PATH_PLUS).float()).tolist()
        walked_traj = torch.nonzero((agent_layer == city.type2label[agent_type]+AGENT_WALKED_PATH_PLUS).float()).tolist()
        goal_agent_pos = torch.nonzero((agent_layer == city.type2label[agent_type]+AGENT_GOAL_PLUS).float()).tolist()
        start_agent_pos = torch.nonzero((agent_layer == city.type2label[agent_type]+AGENT_START_PLUS).float()).tolist()
        # Path
        if len(planned_traj) > 0 and path:  
            if len(planned_traj) > 0 and path:  
                for way_point in planned_traj:
                    vis_grid[int(way_point[0]*scale_factor):int((way_point[0]+1)*scale_factor), \
                        int(way_point[1]*scale_factor):int((way_point[1]+1)*scale_factor)] = agent_color
            if len(walked_traj) > 0 and path:  
                for way_point_ in walked_traj:
                    vis_grid[int(way_point_[0]*scale_factor):int((way_point[0]+1)*scale_factor), \
                        int(way_point_[1]*scale_factor):int((way_point[1]+1)*scale_factor)] = agent_color
            # Points
            if len(start_agent_pos) > 0 and s:
                cv2.drawMarker(vis_grid, (int(start_agent_pos[0][1]*scale_factor), int(start_agent_pos[0][0]*scale_factor)), \
                    agent_color, markerType=cv2.MARKER_TRIANGLE_UP, markerSize=14, thickness=3, line_type=cv2.LINE_AA)
            if len(goal_agent_pos) > 0 and g:   
                cv2.drawMarker(vis_grid, (int(goal_agent_pos[0][1]*scale_factor), int(goal_agent_pos[0][0]*scale_factor)), \
                    agent_color, markerType=cv2.MARKER_TRIANGLE_DOWN, markerSize=14, thickness=3, line_type=cv2.LINE_AA)
            assert len(cur_agent_pos) == 1
            if curr:
                cv2.drawMarker(vis_grid, (int(cur_agent_pos[0][1]*scale_factor), int(cur_agent_pos[0][0]*scale_factor)), \
                    agent_color, markerType=cv2.MARKER_DIAMOND, markerSize=14, thickness=3, line_type=cv2.LINE_AA)

    return vis_grid

def visualize_intersections(intersection_matrix, file_name="intersections.png"):
    # Get unique intersection IDs (excluding 0)
    unique_intersections = np.unique(intersection_matrix)
    unique_intersections = unique_intersections[unique_intersections != 0]
    
    # Create a color map for each intersection ID
    colors = list(mcolors.CSS4_COLORS.values())
    intersection_colors = {uid: colors[i % len(colors)] for i, uid in enumerate(unique_intersections)}

    # Create an RGB visualization matrix
    vis_matrix = np.zeros((*intersection_matrix.shape, 3), dtype=np.uint8)

    for uid, color in intersection_colors.items():
        r, g, b = mcolors.hex2color(color)
        mask = (intersection_matrix == uid)
        vis_matrix[mask] = (np.array([r, g, b]) * 255).astype(np.uint8)

    # Plot
    plt.imshow(vis_matrix)
    plt.title("Intersections Visualization")
    plt.axis('off')
    plt.imsave(file_name, vis_matrix)

def get_pos(local_layer):
    local_layer[local_layer==0] += 0.1
    pos_layer = local_layer == local_layer.astype(np.int64)
    pixels = torch.nonzero(torch.tensor(pos_layer.astype(np.float32)))
    rows = pixels[:, 0]
    cols = pixels[:, 1]
    left = torch.min(cols).item()
    right = torch.max(cols).item()
    top = torch.min(rows).item()
    bottom = torch.max(rows).item()
    return (left, top, right, bottom)

def get_direction(left, left_, top, top_):
    if left_ > left:
        return "right"
    elif left_ < left:
        return "left"
    elif top_ > top:
        return "down"
    elif top_ < top:
        return "up"
    else:
        return "none"

def resize_with_aspect_ratio(image, base_size):
    # Determine the shorter side of the image
    short_side = min(image.shape[:2])
    
    # Calculate the scaling factor
    scale_factor = base_size / short_side
    
    # Calculate the new dimensions of the image
    new_dims = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
    
    # Resize the image with the new dimensions
    resized_img = cv2.resize(image, new_dims, interpolation=cv2.INTER_LINEAR)
    
    return resized_img

def get_street_type(gridmap, position):
    l, t, r, b = position
    partial_grid_horizontal = gridmap[2, t, l-10:l+10]
    if np.sum(partial_grid_horizontal == TYPE_MAP["Mid Lane"]) > 0:
        return "v"
    partial_grid_vertical = gridmap[2, t-10:t+10, l]
    if np.sum(partial_grid_vertical == TYPE_MAP["Mid Lane"]) > 0:
        return "h"
    return None

def rotate_image(image, angle):
    """ Rotate the given image by the specified angle """
    if angle >= 0:
        return image.rotate(angle, expand=True)
    else:
        return image.transpose(Image.FLIP_LEFT_RIGHT)

def gridmap2img_static(gridmap, icon_dict, ego_id):
    # step 1: get the size of the gridmap, create a blank image with size*SCALE
    height, width = gridmap.shape[1], gridmap.shape[2]
    img = np.ones((height*SCALE, width*SCALE, 3), np.uint8) * 255  # assuming white background
    resized_grid = np.repeat(np.repeat(gridmap, SCALE, axis=1), SCALE, axis=2)
    
    # step 2: fill the image with walking street icons
    walking_icon = icon_dict["Walking Street"]
    for i in range(0, img.shape[0], walking_icon.shape[0]):
        for j in range(0, img.shape[1], walking_icon.shape[1]):
            # Calculate the dimensions of the region left in img
            h_space_left = min(walking_icon.shape[0], img.shape[0] - i)
            w_space_left = min(walking_icon.shape[1], img.shape[1] - j)

            # Paste the walking_icon (or its sliced version) to img
            img[i:i+h_space_left, j:j+w_space_left] = walking_icon[:h_space_left, :w_space_left]

    # step 3: read the STREET layer of gridmap, paste the traffic street icons on the traffic street region
    # For the traffic icon
    traffic_icon = icon_dict["Traffic Street"]
    traffic_img = np.zeros_like(img)
    traffic_mask = resized_grid[STREET_ID] == TYPE_MAP["Traffic Street"]
    for i in range(0, img.shape[0], traffic_icon.shape[0]):
        for j in range(0, img.shape[1], traffic_icon.shape[1]):
            # Calculate the dimensions of the region left in img
            h_space_left = min(traffic_icon.shape[0], img.shape[0] - i)
            w_space_left = min(traffic_icon.shape[1], img.shape[1] - j)

            # Paste the walking_icon (or its sliced version) to img
            traffic_img[i:i+h_space_left, j:j+w_space_left] = traffic_icon[:h_space_left, :w_space_left]
    img[traffic_mask] = traffic_img[traffic_mask]

    # For the Overlap and mid lane icon
    traffic_icon = icon_dict["Overlap"]
    traffic_img = np.zeros_like(img)
    traffic_mask = resized_grid[STREET_ID] == TYPE_MAP["Overlap"]
    for i in range(0, img.shape[0], traffic_icon.shape[0]):
        for j in range(0, img.shape[1], traffic_icon.shape[1]):
            # Calculate the dimensions of the region left in img
            h_space_left = min(traffic_icon.shape[0], img.shape[0] - i)
            w_space_left = min(traffic_icon.shape[1], img.shape[1] - j)

            # Paste the walking_icon (or its sliced version) to img
            traffic_img[i:i+h_space_left, j:j+w_space_left] = traffic_icon[:h_space_left, :w_space_left]
    img[traffic_mask] = traffic_img[traffic_mask]

    traffic_img = np.zeros_like(img)
    traffic_mask = resized_grid[STREET_ID] == TYPE_MAP["Mid Lane"]
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            # Paste the walking_icon (or its sliced version) to img
            traffic_img[i:i+h_space_left, j:j+w_space_left] = [255, 215, 0]
    img[traffic_mask] = traffic_img[traffic_mask]

    # For the building icons
    for building in BUILDING_TYPES:
        building_map = resized_grid[BUILDING_ID] == TYPE_MAP[building]
        building_icon = icon_dict[building]
        labeled_matrix, num = label(building_map)

        for i in range(1, num+1):
            local = torch.tensor(labeled_matrix == i)
            pixels = torch.nonzero(local.float())
            rows = pixels[:, 0]
            cols = pixels[:, 1]
            left = torch.min(cols).item()
            right = torch.max(cols).item()
            top = torch.min(rows).item()
            bottom = torch.max(rows).item()
            icon_id = np.random.choice(len(building_icon))
            icon = building_icon[icon_id]
            icon_mask = np.sum(icon > 1, axis=2) > 0
            img[bottom-icon.shape[0]:bottom, left:left+icon.shape[1]][icon_mask] = icon[icon_mask]

    # add ego agent start and goal
    if ego_id > 0:
        ego_map = gridmap[ego_id]
        goal_pos = np.where(ego_map == ego_map.max())
        goal_x, goal_y = goal_pos[0][0]*SCALE, goal_pos[1][0]*SCALE
        int_mask = ego_map == ego_map.astype(np.int64)
        filtered_mask = int_mask * (ego_map != 0)
        start_pos = np.where(filtered_mask)
        start_x, start_y = start_pos[0][0]*SCALE, start_pos[1][0]*SCALE
        cv2.drawMarker(img, (goal_y, goal_x), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)
        cv2.drawMarker(img, (start_y, start_x), (255, 0, 0), markerType=cv2.MARKER_SQUARE, markerSize=15, thickness=2)

    return img

def create_custom_mask(image, threshold=0.1):
    if image.mode == 'RGBA':
        # Use the existing alpha channel
        r, g, b, alpha = image.split()
        alpha = alpha.point(lambda p: 255 if p > threshold else 0)
        return alpha
    else:
        # Create a new mask
        mask = Image.new('L', image.size, 0)  # Start with a fully transparent mask
        pixels = image.load()
        mask_pixels = mask.load()
        
        for i in range(image.size[0]):  # Iterate over width
            for j in range(image.size[1]):  # Iterate over height
                r, g, b = pixels[i, j][:3]
                luminance = int(0.299*r + 0.587*g + 0.114*b)
                if luminance > threshold:
                    mask_pixels[i, j] = 255
        return mask

def paste_car_on_map(map_image, car_image, position, direction, direction_last, type, position_last=None, street_type=None):
    """ Paste car on the map with the correct orientation and position """
    l, t, r, b = position
    if type == "Car":
        # Define rotation angles for directions
        rotation_angles = {
            'up': 0,
            'right': 270,
            'down': 180,
            'left': 90,
            'none': 0
        }
    elif type == "Pedestrian":
        rotation_angles = {
            'up': 0,
            'right': 0,
            'down': -1,
            'left': -1,
            'none': 0
        }
    is_crooked_car = False
    # Rotate the car image based on the direction
    if type == "Pedestrian" or direction == direction_last or direction == "none" or direction_last == "none":
        rotated_car = rotate_image(car_image, rotation_angles[direction])
    else:
        is_crooked_car = True
        bbox_len = int((car_image.width + car_image.height) / np.sqrt(2))
        if (direction == "up" and direction_last == "left") or (direction == "left" and direction_last == "up"):
            rotated_car = rotate_image(car_image, 45)
        elif (direction == "down" and direction_last == "left") or (direction == "left" and direction_last == "down"):
            rotated_car = rotate_image(car_image, 135)
        elif (direction == "down" and direction_last == "right") or (direction == "right" and direction_last == "down"):
            rotated_car = rotate_image(car_image, 225)
        elif (direction == "up" and direction_last == "right") or (direction == "right" and direction_last == "up"):
            rotated_car = rotate_image(car_image, 315)
        else: # this situation should not happen
            rotated_car = rotate_image(car_image, rotation_angles[direction])

    mask = create_custom_mask(rotated_car)

    # Calculate new position after rotation to adjust the car's head position
    if type == "Car":
        if direction == 'up':
            # head position
            if street_type == "v" or street_type is None:
                head_position = ((l+r)//2, t)
                new_position = (head_position[0] - rotated_car.width//2, head_position[1])
            else:
                head_position = ((l+r)//2, b)
                new_position = (head_position[0] - rotated_car.width//2, head_position[1] - rotated_car.height)
        elif direction == 'right':
            if street_type == "h" or street_type is None:
                head_position = (r, (t+b)//2)
                new_position = (head_position[0] - rotated_car.width, head_position[1] - rotated_car.height//2)
            else:
                head_position = (l, (t+b)//2)
                new_position = (head_position[0], head_position[1] - rotated_car.height//2)
        elif direction == 'down':
            if street_type == "v" or street_type is None:
                head_position = ((l+r)//2, b)
                new_position = (head_position[0] - rotated_car.width//2, head_position[1] - rotated_car.height)
            else:
                head_position = ((l+r)//2, t)
                new_position = (head_position[0] - rotated_car.width//2, head_position[1])
        elif direction == 'left':
            if street_type == "h" or street_type is None:
                head_position = (l, (t+b)//2)
                new_position = (head_position[0], head_position[1] - rotated_car.height//2)
            else:
                head_position = (r, (t+b)//2)
                new_position = (head_position[0] - rotated_car.width, head_position[1] - rotated_car.height//2)
        elif direction == 'none':
            if position_last is not None:
                new_position = tuple(position_last)
            else:
                new_position = (l, t)
        if is_crooked_car:
            new_position = (head_position[0] - bbox_len//2, head_position[1] - bbox_len//2)
    elif type == "Pedestrian":
        if direction == "none":
            if position_last is not None:
                new_position = tuple(position_last)
            else:
                new_position = (l, t)
        else:
            center_position = ((l+r)//2, (t+b)//2)
            new_position = (center_position[0] - rotated_car.width//2, center_position[1] - rotated_car.height//2)
    if is_crooked_car:
        icon_bbox_right = new_position[0] + bbox_len
        icon_bbox_bottom = new_position[1] + bbox_len
    else:
        icon_bbox_right = new_position[0] + rotated_car.width
        icon_bbox_bottom = new_position[1] + rotated_car.height
    # Paste the car image onto the map
    map_image.paste(rotated_car, new_position, mask)

    return rotated_car, map_image, list(new_position), (new_position[0], new_position[1], icon_bbox_right, icon_bbox_bottom)

def gridmap2img_agents(vis_dataset, agent_next_actions, step_name, gridmap, gridmap_, icon_dict, static_map, last_icons=None, agents=None):
    current_map = static_map.copy()
    current_map = Image.fromarray(current_map)
    agent_layer = gridmap[BASIC_LAYER:]
    resized_grid = np.repeat(np.repeat(agent_layer, SCALE, axis=1), SCALE, axis=2)
    agent_layer_ = gridmap_[BASIC_LAYER:]
    resized_grid_ = np.repeat(np.repeat(agent_layer_, SCALE, axis=1), SCALE, axis=2)
    icon_dict_local = {
        "icon": {},
        "pos": {}
    }
    cur_step = int(step_name.split("_")[1][4:])
    last_step_name = "{}{:0>4d}".format(step_name[:-4], cur_step-1)

    for i in range(resized_grid.shape[0]):
        local_layer = resized_grid[i]
        left, top, right, bottom = get_pos(local_layer)
        local_layer_ = resized_grid_[i]
        left_, top_, right_, bottom_ = get_pos(local_layer_)
        direction = get_direction(left, left_, top, top_)
        if cur_step == 1:
            direction_last = direction
        else:
            direction_last = vis_dataset[last_step_name]["Directions"][i]
        pos = (left, top, right, bottom)
        
        agent_type = LABEL_MAP[local_layer[top, left].item()]     
        agent_name = "{}_{}".format(agent_type, BASIC_LAYER + i)
        if agents != None:
            vis_dataset[step_name]["Types"][i] = agent_type
            concepts = agents[agent_name]["concepts"]
            vis_dataset[step_name]["Priorities"][i] = concepts["priority"]
            # Get detailed type of agent
            agent_detailed_type = list(set(list(DETAILED_TYPE_MAP.keys())) \
                                    & set(list(concepts.keys())))
            if not agent_detailed_type:
                if agent_type == "Car":
                    agent_detailed_type = "normal_car"
                else:
                    agent_detailed_type = "normal_pedestrian"
            else:
                agent_detailed_type = agent_detailed_type[0]
            vis_dataset[step_name]["Detailed_types"][i] = agent_detailed_type
            vis_dataset[step_name]["Directions"][i] = direction
            is_ambulance = False
            is_police = False
            is_young = False
            is_bus = False
            is_tiro = False
            is_reckless = False
            is_old = False
            if "tiro" in concepts.keys():
                if concepts["tiro"] == 1.0:
                    is_tiro = True
            if "bus" in concepts.keys():
                if concepts["bus"] == 1.0:
                    is_bus = True
            if "ambulance" in concepts.keys():
                if concepts["ambulance"] == 1.0:
                    is_ambulance = True
            if "old" in concepts.keys():
                if concepts["old"] == 1.0:
                    is_old = True
            if "young" in concepts.keys():
                if concepts["young"] == 1.0:
                    is_young = True
            if "police" in concepts.keys():
                if concepts["police"] == 1.0:
                    is_police = True
            if "reckless" in concepts.keys():
                if concepts["reckless"] == 1.0:
                    is_reckless = True
            if is_ambulance:
                icon_list = icon_dict["Ambulance"]
                icon_id = i%len(icon_list)
                icon = icon_list[icon_id]
            elif is_bus:
                icon_list = icon_dict["Bus"]
                icon_id = i%len(icon_list)
                icon = icon_list[icon_id]
            elif is_tiro:
                icon_list = icon_dict["Tiro"]
                icon_id = i%len(icon_list)
                icon = icon_list[icon_id]
            elif is_old:
                icon_list = icon_dict["Pedestrian_old"]
                icon_id = i%len(icon_list)
                icon = icon_list[icon_id]
            elif is_young:
                icon_list = icon_dict["Pedestrian_young"]
                icon_id = i%len(icon_list)
                icon = icon_list[icon_id]
            elif is_police:
                icon_list = icon_dict["Police"]
                icon_id = i%len(icon_list)
                icon = icon_list[icon_id]
            elif is_reckless:
                icon_list = icon_dict["Reckless"]
                icon_id = i%len(icon_list)
                icon = icon_list[icon_id]
            else:
                if agent_type == "Pedestrian":
                    icon_list = icon_dict["Pedestrian"]
                    icon_id = i%len(icon_list)
                    icon = icon_list[icon_id]
                if agent_type == "Car":
                    icon_list = icon_dict["Car"]
                    icon_id = i%len(icon_list)
                    icon = icon_list[icon_id]
        else:
            icon_list = icon_dict[agent_type]
            icon_id = i%len(icon_list)
            icon = icon_list[icon_id]

        if agent_type == "Car":
            street_type = get_street_type(resized_grid, pos)
        else:
            street_type = None    

        if last_icons is not None:
            if direction == "none":
                icon = last_icons["icon"]["{}_{}".format(agent_type, i)][1]
                position = last_icons["pos"]["{}_{}".format(agent_type, i)]
                icon, current_map, last_position, bbox = paste_car_on_map(current_map, icon, pos, direction, direction_last, agent_type, position_last=position, street_type=street_type)
            else:
                icon = last_icons["icon"]["{}_{}".format(agent_type, i)][0]
                icon, current_map, last_position, bbox = paste_car_on_map(current_map, icon, pos, direction, direction_last, agent_type, street_type=street_type)
            last_icons["icon"]["{}_{}".format(agent_type, i)][1] = icon
            last_icons["pos"]["{}_{}".format(agent_type, i)] = last_position
        else:
            icon_img = Image.fromarray(icon) 
            icon_dict_local["icon"]["{}_{}".format(agent_type, i)] = [icon_img]
            current_icon, current_map, last_position, bbox = paste_car_on_map(current_map, icon_img, pos, direction, direction_last, agent_type, street_type=street_type)
            icon_dict_local["icon"]["{}_{}".format(agent_type, i)].append(current_icon)
            icon_dict_local["pos"]["{}_{}".format(agent_type, i)] = last_position

        vis_dataset[step_name]["Bboxes"][i] = (max(bbox[0],0), max(bbox[1],0), min(bbox[2],current_map.size[0]), min(bbox[3],current_map.size[1]))
        vis_dataset[step_name]["Next_actions"][i] = agent_next_actions[i]

    if last_icons is not None:
        return current_map, last_icons, vis_dataset
    else:
        return current_map, icon_dict_local, vis_dataset

def pkl2city_imgs(cached_observation, vis_dataset, world_idx, icon_dir_dict, output_folder, ego_id=-1, crop=[0, 1024, 0, 1024]):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    obs = cached_observation["Time_Obs"]
    agents = cached_observation["Static Info"]["Agents"]

    print(obs.keys())
    time_steps = list(obs.keys())
    time_steps.sort()

    icon_dict = get_random_icon_dict(icon_dir_dict, True) # sample icon from icon lib

    static_map = gridmap2img_static(obs[time_steps[0]]["World"].numpy(), icon_dict, ego_id)
    # static_map_img.save("{}/static_layout.png".format(output_folder))
    last_icons = None
    for key in trange(time_steps[0], time_steps[-1]):
        step_name = "World{}_step{:0>4d}".format(world_idx, key)
        vis_dataset[step_name] = {}
        vis_dataset[step_name]["Image_path"] = os.path.join(output_folder ,"step_{:0>4d}.png".format(key))
        vis_dataset[step_name]["Predicate_groundings"] = obs[key]["Predicate_groundings"]
        vis_dataset[step_name]["Bboxes"] = {}
        vis_dataset[step_name]["Types"] = {}
        vis_dataset[step_name]["Detailed_types"] = {}
        vis_dataset[step_name]["Priorities"] = {}
        vis_dataset[step_name]["Directions"] = {}
        vis_dataset[step_name]["Next_actions"] = {}
        grid = obs[key]["World"].numpy()
        grid_ = obs[key+1]["World"].numpy()
        icon_dict = get_random_icon_dict(icon_dir_dict) # sample icon from icon lib
        img, last_icons, vis_dataset = gridmap2img_agents(vis_dataset, list(obs[key]["Agent_actions"].values()), step_name, grid, grid_, icon_dict, static_map, last_icons, agents)
        last_icons = get_random_last_icons(last_icons, vis_dataset[step_name]["Detailed_types"], icon_dict)
        xmin, xmax, ymin, ymax = crop
        img = img.crop((xmin, ymin, xmax, ymax))
        # Save the image
        output_path = "{}/step_{:0>4d}.png".format(output_folder, key)
        img.save(output_path)

    return vis_dataset

def get_random_icon_dict(icon_dir_dict, static = False):
    icon_dict = {}
    print("Generating random icons for each agent type...")
    s = time.time()
    for key in icon_dir_dict.keys():
        icon_dir_path = icon_dir_dict[key]["icon_dir_path"]
        icon_num = icon_dir_dict[key]["icon_num"]
        if key in ["House", "Office", "Store", "Gas Station", "Garage"]:
            if static:
                icon_path_list = []
                for _ in range(10):
                    icon_idx = np.random.choice(icon_num)
                    # option 1 (faster, for imgs with formulated names)
                    # icon_path_list.append(os.path.join(icon_dir_path, "image_{}.png".format(icon_idx)))
                    # option 2 (slower, for imgs with random names)
                    icon_path_list.append(os.path.join(icon_dir_path, os.listdir(icon_dir_path)[icon_idx]))
                raw_img = [remove_background_alpha_channel(path) for path in icon_path_list]
                resized_img = [resize_with_aspect_ratio(img, ICON_SIZE_DICT[key]) for img in raw_img]
                icon_dict[key] = resized_img
            else:
                icon_dict[key] = [None]
        elif key in ["Car", "Pedestrian", "Ambulance", "Bus", "Tiro", "Police", "Reckless", "Pedestrian_old", "Pedestrian_young"]:
            icon_path_list = []
            for _ in range(3):
                icon_idx = np.random.choice(icon_num)
                # option 1 (faster, for imgs with formulated names)
                # icon_path_list.append(os.path.join(icon_dir_path, "image_{}.png".format(icon_idx)))
                # option 2 (slower, for imgs with random names)
                icon_path_list.append(os.path.join(icon_dir_path, os.listdir(icon_dir_path)[icon_idx]))
            raw_img = [remove_background_alpha_channel(path) for path in icon_path_list]
            resized_img = [resize_with_aspect_ratio(img, ICON_SIZE_DICT[key]) for img in raw_img]
            # check and convert [255, 255, 255] to [0, 0, 0]
            icon_dict[key] = resized_img
        else:
            if static:
                icon_idx = np.random.choice(icon_num)
                # option 1 (faster, for imgs with formulated names)
                # icon_path = os.path.join(icon_dir_path, "image_{}.png".format(icon_idx))
                # option 2 (slower, for imgs with random names)
                icon_path = os.path.join(icon_dir_path, os.listdir(icon_dir_path)[icon_idx])
                raw_img = cv2.cvtColor(cv2.imread(icon_path), cv2.COLOR_BGR2RGB)
                resized_img = resize_with_aspect_ratio(raw_img, ICON_SIZE_DICT[key])
                icon_dict[key] = resized_img
    e = time.time()
    print("Time taken to generate random icons: {:.2f}s".format(e-s))
    return icon_dict

def remove_background_alpha_channel(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # Make sure to read the alpha channel if present
    assert image is not None, "Image not found at path: {}".format(image_path)
    if image.shape[2] == 4:  # Check if there is an alpha channel
        # Use the alpha channel as a mask to set background pixels to black
        # Here we assume background is where alpha value is 0
        mask = image[:, :, 3] == 0
        image[mask, :3] = [0, 0, 0]  # Set color channels to black where mask is True
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)  # Convert to RGB
    return image  # Return only the RGB channels


def get_random_last_icons(last_icons, detailed_types, icon_dict):
    for i, agent_name in enumerate(list(last_icons["icon"].keys())):
        detailed_type = detailed_types[i]
        if detailed_type == "normal_car":
            new_icon_list = icon_dict["Car"]
            icon_id = i%len(new_icon_list)
            new_icon = new_icon_list[icon_id]
        elif detailed_type == "normal_pedestrian":
            new_icon_list = icon_dict["Pedestrian"]
            icon_id = i%len(new_icon_list)
            new_icon = new_icon_list[icon_id]
        else:  
            new_icon_list = icon_dict[DETAILED_TYPE_MAP[detailed_type]]
            icon_id = i%len(new_icon_list)
            new_icon = new_icon_list[icon_id]
        last_icons["icon"][agent_name][0] = Image.fromarray(new_icon)
    return last_icons
