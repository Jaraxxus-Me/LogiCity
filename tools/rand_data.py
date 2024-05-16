import numpy as np
import random
import pickle as pkl

data_path = "vis_dataset/hard_fixed_final/train/train_hard_fixed_final.pkl"

with open(data_path, 'rb') as f:
    raw_data = pkl.load(f)

np.random.seed(2)
# raw_data is a dictionary
key_list = list(raw_data.keys())
np.random.shuffle(key_list)
rand_data = {}
for key in key_list:
    rand_data[key] = raw_data[key]

rand_data_path = "vis_dataset/hard_fixed_final/train/train_hard_fixed_final_rand2.pkl"
with open(rand_data_path, 'wb') as f:
    pkl.dump(rand_data, f)
    print(f"Randomized data saved to {rand_data_path}")

# np.random.seed(1)
# np.random.shuffle(key_list)
# rand_data = {}
# for key in key_list:
#     rand_data[key] = raw_data[key]

# rand_data_path = "vis_dataset/hard_random_final/train/train_hard_random_final_rand1.pkl"
# with open(rand_data_path, 'wb') as f:
#     pkl.dump(rand_data, f)
#     print(f"Randomized data saved to {rand_data_path}")
