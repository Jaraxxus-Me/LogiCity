import yaml
import numpy as np
from logicity.core.config import *

def list_representer(dumper, data):
    return dumper.represent_sequence(u'tag:yaml.org,2002:seq', data, flow_style=False)

yaml.add_representer(list, list_representer)
np.random.seed(42)

base_yaml_file = 'config/maps/v1.0.yaml'

with open(base_yaml_file, 'r') as file:
    city_config = yaml.safe_load(file)

city_config['buildings'] = []
city_config['streets'] = []
# Traffic Streets
for j in range(0, WORLD_SIZE, (BUILDING_SIZE*2+WALKING_STREET_WID*3+TRAFFIC_STREET_WID)):
    # vertical
    traf_stree = {
        'position': [0, j],
        'length': WORLD_SIZE,
        'orientation': 'vertical',
        'type': 'Traffic Street',
        'directions': 2,
        'width': TRAFFIC_STREET_WID
    }
    city_config['streets'].append(traf_stree)
    # horison
    traf_stree = {
        'position': [j, 0],
        'length': WORLD_SIZE,
        'orientation': 'horizontal',
        'type': 'Traffic Street',
        'directions': 2,
        'width': TRAFFIC_STREET_WID
    }
    city_config['streets'].append(traf_stree)

# Walking
for j in range(TRAFFIC_STREET_WID, WORLD_SIZE, (BUILDING_SIZE*2+WALKING_STREET_WID*3+TRAFFIC_STREET_WID)):
    # vertical
    traf_stree = {
        'position': [TRAFFIC_STREET_WID, j],
        'length': WORLD_SIZE-2*TRAFFIC_STREET_WID,
        'orientation': 'vertical',
        'type': 'Walking Street',
        'directions': 2,
        'width': WALKING_STREET_WID
    }
    city_config['streets'].append(traf_stree)
    traf_stree = {
        'position': [TRAFFIC_STREET_WID, j+BUILDING_SIZE*2+WALKING_STREET_WID*2],
        'length': WORLD_SIZE-2*TRAFFIC_STREET_WID,
        'orientation': 'vertical',
        'type': 'Walking Street',
        'directions': 2,
        'width': WALKING_STREET_WID
    }
    city_config['streets'].append(traf_stree)
    # horison
    traf_stree = {
        'position': [j, TRAFFIC_STREET_WID],
        'length': WORLD_SIZE-2*TRAFFIC_STREET_WID,
        'orientation': 'horizontal',
        'type': 'Walking Street',
        'directions': 2,
        'width': WALKING_STREET_WID
    }
    city_config['streets'].append(traf_stree)
    traf_stree = {
        'position': [j+BUILDING_SIZE*2+WALKING_STREET_WID*2, TRAFFIC_STREET_WID],
        'length': WORLD_SIZE-2*TRAFFIC_STREET_WID,
        'orientation': 'horizontal',
        'type': 'Walking Street',
        'directions': 2,
        'width': WALKING_STREET_WID
    }
    city_config['streets'].append(traf_stree)

# inter walking and buildings
n = np.sqrt(NUM_OF_BLOCKS).astype(np.int16)

for i in range(n):
    for j in range(n):
        # buildings
        block_id = i*5 + j + 1
        size = BUILDING_SIZE
        # vertical
        vs = WALKING_STREET_WID+TRAFFIC_STREET_WID + BUILDING_SIZE + j*(BUILDING_SIZE*2+WALKING_STREET_WID*3+TRAFFIC_STREET_WID)
        hs = WALKING_STREET_WID+TRAFFIC_STREET_WID + BUILDING_SIZE + i*(BUILDING_SIZE*2+WALKING_STREET_WID*3+TRAFFIC_STREET_WID)
        traf_stree = {
        'position': [WALKING_STREET_WID+TRAFFIC_STREET_WID + i*(BUILDING_SIZE*2+WALKING_STREET_WID*3+TRAFFIC_STREET_WID), vs],
        'length': WALKING_STREET_LENGTH,
        'orientation': 'vertical',
        'type': 'Walking Street',
        'directions': 2,
        'width': WALKING_STREET_WID
        }
        city_config['streets'].append(traf_stree)

        traf_stree = {
        'position': [hs, WALKING_STREET_WID+TRAFFIC_STREET_WID + j*(BUILDING_SIZE*2+WALKING_STREET_WID*3+TRAFFIC_STREET_WID)],
        'length': WALKING_STREET_LENGTH,
        'orientation': 'horizontal',
        'type': 'Walking Street',
        'directions': 2,
        'width': WALKING_STREET_WID
        }
        city_config['streets'].append(traf_stree)

        chosen_building = np.random.choice(BUILDING_TYPES, p=BUILDING_PROB)
        chosen_building = str(chosen_building)
        # top-left
        building = {
            'position': [hs - size, vs - size],
            'size': [size, size],
            'block': block_id,
            'type': chosen_building,
            'height': np.random.randint(10, 30)
        }
        city_config['buildings'].append(building)

        chosen_building = np.random.choice(BUILDING_TYPES, p=BUILDING_PROB)
        chosen_building = str(chosen_building)
        # top-right
        building = {
            'position': [hs - size, vs + WALKING_STREET_WID],
            'size': [size, size],
            'block': block_id,
            'type': chosen_building,
            'height': np.random.randint(10, 30)
        }
        city_config['buildings'].append(building)

        chosen_building = np.random.choice(BUILDING_TYPES, p=BUILDING_PROB)
        chosen_building = str(chosen_building)
        # bottom-left
        building = {
            'position': [hs + WALKING_STREET_WID, vs - size],
            'size': [size, size],
            'block': block_id,
            'type': chosen_building,
            'height': np.random.randint(10, 30)
        }
        city_config['buildings'].append(building)

        chosen_building = np.random.choice(BUILDING_TYPES, p=BUILDING_PROB)
        chosen_building = str(chosen_building)
        # bottom-right
        building = {
            'position': [hs + WALKING_STREET_WID, vs + WALKING_STREET_WID],
            'size': [size, size],
            'block': block_id,
            'type': chosen_building,
            'height': np.random.randint(10, 30)
        }
        city_config['buildings'].append(building)


with open('TEST.yaml', 'w') as file:
    yaml.dump(city_config, file, default_flow_style=False)


