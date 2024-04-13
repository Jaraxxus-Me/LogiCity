import numpy as np
import torch
import pickle as pkl
from PIL import Image

# data loader class for data buffer inherited from torch.utils.data.Dataset
class WMDataset(torch.utils.data.Dataset):
    def __init__(self, data_buffer, batch_size=256, shuffle=True):
        self.data_buffer = data_buffer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(self.data_buffer["obs"])
        self.indices = np.arange(self.data_size)
        self.num_batches = self.data_size // self.batch_size
        self.batch_idx = 0
        self.reward_val = np.array([-1, 0, 1, 10, -10])
        self.reset()
    
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
        idx = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        obs = np.stack(self.data_buffer["obs"])[idx]
        acts = np.stack(self.data_buffer["acts"])[idx]
        # rews = np.array([np.where(self.reward_val == self.data_buffer["rews"][i]) for i in idx], dtype=np.int32).reshape(-1, 1)
        # transform rews to one-hot vector
        rews = np.zeros((len(idx), len(self.reward_val)))
        # print([self.data_buffer["rews"][i] for i in idx])
        try: 
            rews[np.arange(len(idx)), np.array([np.where(self.reward_val == self.data_buffer["rews"][i]) for i in idx]).reshape(-1)] = 1
        except: 
            pass
            # print(len([self.data_buffer["rews"][i] for i in idx]))
            # input()
        # rews = np.array(np.stack(self.data_buffer["rews"])[idx])
        dones = np.array(np.stack(self.data_buffer["dones"])[idx], dtype=np.int32).reshape(-1, 1)
        next_obs = np.stack(self.data_buffer["next_obs"])[idx]
        return obs, acts, rews, dones, next_obs
    
    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.batch_idx = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.batch_idx < self.num_batches:
            batch = self.__getitem__(self.batch_idx)
            self.batch_idx += 1
            return batch
        else:
            self.reset()
            raise StopIteration

class VisDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            vis_dataset_path, 
            batch_size=1, 
            shuffle=True,
        ):

        self.batch_size = batch_size
        self.shuffle = shuffle
        with open(vis_dataset_path, "rb") as f:
            self.vis_dataset = pkl.load(f)
        self.vis_dataset_list = list(self.vis_dataset.keys())
        if self.shuffle:
            np.random.shuffle(self.vis_dataset_list)
        self.data_size = len(self.vis_dataset_list)
        self.num_batches = self.data_size // self.batch_size

    def read_img(self, filepath):
        img = Image.open(filepath)
        img = np.array(img)
        return img

    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
        step_names = self.vis_dataset_list[idx*self.batch_size:(idx+1)*self.batch_size]
        imgs = []
        bboxes = []
        types = []
        next_actions = []
        for step_name in step_names:
            imgs.append(self.read_img(self.vis_dataset[step_name]["Image_path"]))
            bboxes.append(list(self.vis_dataset[step_name]["Bboxes"].values()))
            types.append(list(self.vis_dataset[step_name]["Types"].values()))
            next_actions.append(list(self.vis_dataset[step_name]["Next_actions"].values()))

        imgs = torch.Tensor(np.array(imgs))
        bboxes = torch.Tensor(np.array(bboxes))
        next_actions = torch.Tensor(np.array(next_actions))

        out_dict = {
            "step_names": step_names,
            "imgs": imgs,
            "bboxes": bboxes,
            "types": types,
            "next_actions": next_actions,
        }

        return out_dict

if __name__ == '__main__': 
    import joblib
    data = joblib.load("log/expert_all.pkl")
    
    dataset = WMDataset(data)
    for obs, acts, rews, dones, next_obs in dataset:
        print(obs.shape, acts.shape, rews.shape, dones.shape, next_obs.shape)
        print(rews[0])
        break