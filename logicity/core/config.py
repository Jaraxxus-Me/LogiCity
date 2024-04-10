# City
BUILDING_TYPES = ["House", "Office", "Gas Station", "Garage", "Store"]
BUILDING_PROB = [0.2, 0.2, 0.05, 0.15, 0.4]
LABEL_MAP = {
    -1: 'Overlap',
    0: 'Under Construction',
    1: 'Walking Street',
    2: 'Traffic Street',
    2.5: 'Mid Lane',
    3: 'House',
    4: 'Gas Station',
    5: 'Office',
    6: 'Garage',
    7: 'Store',
    8: 'Pedestrian',
    9: 'Car'
}
TYPE_MAP = {v: k for k, v in LABEL_MAP.items()}
COLOR_MAP = {
            -1: [100, 100, 100],
            0: [200, 200, 200],       # Grey for empty
            1: [152, 216, 170],           # Green for walking street
            2: [168, 161, 150],        # Red for traffic street
            2.5: [0, 215, 255],        # Red for traffic street
            3: [255, 204, 112],       # house
            4: [34, 102, 141],        # gas station
            5: [255, 250, 221],       # office
            6: [142, 205, 221],       # garage
            7: [255, 63, 164]         # store
        }
# 0: block, 1: building, 2: street, 3+: agent
BUILDING_SIZE=10
BLOCK_ID = 0
BUILDING_ID = 1
STREET_ID = 2
BASIC_LAYER = 3
INTERSECTION_CODE = -1
MID_LINE_CODE_PLUS = 0.5
NUM_OF_BLOCKS = 25
WALKING_STREET_WID = 5
TRAFFIC_STREET_WID = 11
NUM_INTERSECTIONS_BLOCKS = 32
NUM_INTERSECTIONS_LINES = 80
WORLD_SIZE = TRAFFIC_STREET_WID*6 + (BUILDING_SIZE*2 + WALKING_STREET_WID*3)*5
WALKING_STREET_LENGTH = BUILDING_SIZE*2 + WALKING_STREET_WID
TRAFFIC_STREET_LENGTH = WALKING_STREET_LENGTH

# Agents
REACH_GOAL_WAITING = 1
AGENT_TYPES = ["Pedestrian", "Car"]
AGENT_GLOBAL_PATH_PLUS = 0.1
AGENT_WALKED_PATH_PLUS = -0.1
AGENT_GOAL_PLUS = 0.3
AGENT_START_PLUS = -0.2

A_START_E = 1.0
AT_INTERSECTION_E = 1

PEDES_GOAL_START =  ["House", "Office"]
# should be odd
PED_GOAL_START_INCLUDE_KERNEL = 3
PED_GOAL_START_EXCLUDE_KERNEL = WALKING_STREET_WID+2

CAR_STREET_OFFSET = 2
CAR_GOAL_START = ["Gas Station", "Garage", "Store"]
# should be odd
CAR_GOAL_START_INCLUDE_KERNEL = (WALKING_STREET_WID+4)*2+1
CAR_GOAL_START_EXCLUDE_KERNEL = WALKING_STREET_WID + TRAFFIC_STREET_WID//2 + 1
CAR_SIZE = 3 # kernel to enlarge car occupency
OCC_CHECK_RANGE = {
    "Pedestrian": 3,
    "Car": 8
}
OCC_CHECK_ANGEL = 0.1
DIRECTION_VECTOR = {
    "Left": (0, -1),
    "Right": (0, 1),
    "Up": (-1, 0),
    "Down": (1, 0)
}
PED_AGGR = 0.5

# Bus Routes, got these from midline segments
ROAD_GRAPH_NODES = 'logicity/core/road_graph.txt'
BUS_ROUTES = {
    "61A": [24, 25, 28, 29, 32, 33, 204, 205, 208, 209, 212, 213, 94, 95, 90, 91, 176, 177, 106, 107, 158, 159, 154, 155, 150, 151, 146, 147, 24]
}

# Rules

CLOSE_RANGE_MIN = 12
CLOSE_RANGE_MAX = 20
# solver processer
NUM_PROCESS = 4
AGENT_FOV = 25 # better be larger than intersection width
ACTION2NAME = {
    0: "Left",
    1: "Right",
    2: "Up",
    3: "Down",
    4: "Stop"
}
