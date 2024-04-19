import os
import numpy as np
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


DETAILED_TYPE_MAP = {
    "tiro": "Tiro", 
    "bus": "Bus", 
    "ambulance": "Ambulance", 
    "old": "Pedestrian_old", 
    "young": "Pedestrian_young", 
    "police": "Police", 
    "reckless": "Reckless"
}

IMAGE_BASE_PATH = "./imgs"

PATH_DICT = {
    "Car": [os.path.join(IMAGE_BASE_PATH, "car{}.png").format(i) for i in range(1, 3)],
    "Ambulance": os.path.join(IMAGE_BASE_PATH, "car_ambulance.png"),
    "Bus": os.path.join(IMAGE_BASE_PATH, "car_bus.png"),
    "Tiro": os.path.join(IMAGE_BASE_PATH, "car_tiro.png"),
    "Police": os.path.join(IMAGE_BASE_PATH, "car_police.png"),
    "Reckless": os.path.join(IMAGE_BASE_PATH, "car_reckless.png"),
    "Pedestrian": [os.path.join(IMAGE_BASE_PATH, "pedestrian{}.png").format(i) for i in range(1, 4)],
    "Pedestrian_old": os.path.join(IMAGE_BASE_PATH, "pedestrian_old.png"),
    "Pedestrian_young": os.path.join(IMAGE_BASE_PATH, "pedestrian_young.png"),
    "Walking Street": os.path.join(IMAGE_BASE_PATH, "walking.png"),
    "Traffic Street": os.path.join(IMAGE_BASE_PATH, "traffic.png"),
    "Overlap": os.path.join(IMAGE_BASE_PATH, "crossing.png"),
    "Gas Station": os.path.join(IMAGE_BASE_PATH, "gas.png"),
    "Garage": os.path.join(IMAGE_BASE_PATH, "garage.png"),
    "House": [os.path.join(IMAGE_BASE_PATH, "house{}.png").format(i) for i in range(1, 4)],
    "Office": [os.path.join(IMAGE_BASE_PATH, "office{}.png").format(i) for i in range(1, 4)],
    "Store": [os.path.join(IMAGE_BASE_PATH, "store{}.png").format(i) for i in range(1, 4)],
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
            if building in ["House", "Office", "Store"]:
                icon_id = np.random.choice(3)
                icon = building_icon[icon_id]
            else:
                icon = building_icon
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

def paste_car_on_map(map_image, car_image, position, direction, type, position_last=None, street_type=None):
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


    # Rotate the car image based on the direction
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
    elif type == "Pedestrian":
        if direction == "none":
            if position_last is not None:
                new_position = tuple(position_last)
            else:
                new_position = (l, t)
        else:
            center_position = ((l+r)//2, (t+b)//2)
            new_position = (center_position[0] - rotated_car.width//2, center_position[1] - rotated_car.height//2)
    

    # Paste the car image onto the map
    map_image.paste(rotated_car, new_position, mask)

    return rotated_car, map_image, list(new_position)

def gridmap2img_agents(gridmap, gridmap_, icon_dict, static_map, last_icons=None, agents=None):
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

    for i in range(resized_grid.shape[0]):
        local_layer = resized_grid[i]
        left, top, right, bottom = get_pos(local_layer)
        local_layer_ = resized_grid_[i]
        left_, top_, right_, bottom_ = get_pos(local_layer_)
        direction = get_direction(left, left_, top, top_)
        pos = (left, top, right, bottom)
        
        agent_type = LABEL_MAP[local_layer[top, left].item()]     
        agent_name = "{}_{}".format(agent_type, BASIC_LAYER + i)
        if agents != None:
            concepts = agents[agent_name]["concepts"]
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
                icon = icon_dict["Ambulance"]
            elif is_bus:
                icon = icon_dict["Bus"]
            elif is_tiro:
                icon = icon_dict["Tiro"]
            elif is_old:
                icon = icon_dict["Pedestrian_old"]
            elif is_young:
                icon = icon_dict["Pedestrian_young"]
            elif is_police:
                icon = icon_dict["Police"]
            elif is_reckless:
                icon = icon_dict["Reckless"]
            else:
                if agent_type == "Pedestrian":
                    icon_list = icon_dict[agent_type].copy()
                    icon_id = i%len(icon_list)
                    icon = icon_list[icon_id]
                if agent_type == "Car":
                    icon_list = icon_dict[agent_type].copy()
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
                icon, current_map, last_position = paste_car_on_map(current_map, icon, pos, direction, agent_type, position_last=position, street_type=street_type)
            else:
                icon = last_icons["icon"]["{}_{}".format(agent_type, i)][0]
                icon, current_map, last_position = paste_car_on_map(current_map, icon, pos, direction, agent_type, street_type=street_type)
            last_icons["icon"]["{}_{}".format(agent_type, i)][1] = icon
            last_icons["pos"]["{}_{}".format(agent_type, i)] = last_position
        else:
            icon_img = Image.fromarray(icon) 
            icon_dict_local["icon"]["{}_{}".format(agent_type, i)] = [icon_img]
            current_icon, current_map, last_position = paste_car_on_map(current_map, icon_img, pos, direction, agent_type, street_type=street_type)
            icon_dict_local["icon"]["{}_{}".format(agent_type, i)].append(current_icon)
            icon_dict_local["pos"]["{}_{}".format(agent_type, i)] = last_position

    if last_icons is not None:
        return current_map, last_icons
    else:
        return current_map, icon_dict_local

def pkl2city_imgs(pkl_path, output_folder, ego_id=-1, crop=[0, 1024, 0, 1024]):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
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

    with open(pkl_path, "rb") as f:
        data = pkl.load(f)
        obs = data["Time_Obs"]
        agents = data["Static Info"]["Agents"]

    print(obs.keys())
    time_steps = list(obs.keys())
    time_steps.sort()
    static_map = gridmap2img_static(obs[time_steps[0]]["World"].numpy(), icon_dict, ego_id)
    static_map_img = Image.fromarray(static_map)
    static_map_img.save("{}/static_layout.png".format(output_folder))
    last_icons = None
    for key in trange(time_steps[0], time_steps[-1]):
        grid = obs[key]["World"].numpy()
        grid_ = obs[key+1]["World"].numpy()
        img, last_icons = gridmap2img_agents(grid, grid_, icon_dict, static_map, last_icons, agents)
        xmin, xmax, ymin, ymax = crop
        img = img.crop((xmin, ymin, xmax, ymax))
        # Define the text to be added
        text = "#{}".format(key)

        # Specify the position for the text (x, y coordinates)
        position = (10, 10)  # 10 pixels from the left and 30 from the top

        # Create an ImageDraw object
        draw = ImageDraw.Draw(img)

        # Define font type and size (you might need to provide the path to a .ttf font file)
        try:
            font = ImageFont.truetype("arial.ttf", size=100)  # Example font, adjust the path and size as needed
        except IOError:
            font = ImageFont.load_default()

        # Define text color
        color = (255, 255, 255)  # White color

        # Add text to image
        draw.text(position, text, fill=color, font=font)

        # Save the image
        output_path = "{}/step_{:0>4d}.png".format(output_folder, key)
        img.save(output_path)
    # cv2.destroyAllWindows()

    return

