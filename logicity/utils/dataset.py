import numpy as np
import torch
import pickle as pkl
from PIL import Image, ImageDraw, ImageFont
import os
import cv2


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
            gt_vis=False,
        ):

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.gt_vis = gt_vis
        with open(vis_dataset_path, "rb") as f:
            self.vis_dataset = pkl.load(f)
        self.vis_dataset_list = list(self.vis_dataset.keys())
        if self.shuffle:
            np.random.shuffle(self.vis_dataset_list)
        self.data_size = len(self.vis_dataset_list)
        self.num_batches = self.data_size // self.batch_size
        self.direction_dict = {
            "left": [1, 0, 0, 0],
            "right": [0, 1, 0, 0],
            "up": [0, 0, 1, 0],
            "down": [0, 0, 0, 1],
            "none": [0, 0, 0, 0],
        }

    def read_img(self, filepath):
        img = Image.open(filepath)
        img = np.array(img)
        return img
    
    def gt_visualization(self, step_name, img_path, predicates, bboxes, detailed_types, output_folder):
        im = Image.open(img_path)
        draw = ImageDraw.Draw(im)
        # Define font type and size (you might need to provide the path to a .ttf font file)
        try:
            font = ImageFont.truetype("arial.ttf", size=100)  # Example font, adjust the path and size as needed
        except IOError:
            font = ImageFont.load_default()
        for i, bbox in enumerate(bboxes):
            label = detailed_types[i]
            font_bbox = font.getbbox(label)
            text_origin = [bbox[0], bbox[1] - (font_bbox[3]-font_bbox[1])]
            label_bbox = draw.textbbox(text_origin, label, font)
            draw.rectangle(bbox, outline='red',width=2)
            draw.rectangle(label_bbox, fill='red')
            draw.text(text_origin, label, fill=(255, 255, 255), font=font)

        na = np.array(im)
        agent_num = len(bboxes)
        bbox_centers = []
        for i in range(agent_num):
            bbox_centers.append([(bboxes[i][0]+bboxes[i][2])//2, (bboxes[i][1]+bboxes[i][3])//2])
        for i in range(agent_num):
            for j in range(agent_num):
                if i == j:
                    continue
                p_str = ""
                for p_name in predicates.keys():
                    if "Is" not in p_name and predicates[p_name][i][j]:
                        p_str += p_name + ", "
                if len(p_str) != 0:
                    if i > j:
                        start_pos = (bbox_centers[i][0]-5, bbox_centers[i][1]-5)
                        end_pos = (bbox_centers[j][0]-5, bbox_centers[j][1]-5)
                    else:
                        start_pos = (bbox_centers[i][0]+5, bbox_centers[i][1]+5)
                        end_pos = (bbox_centers[j][0]+5, bbox_centers[j][1]+5)
                    text_pos = ((start_pos[0]*2//3+end_pos[0]//3)-10, (start_pos[1]*2//3+end_pos[1]//3))
                    cv2.arrowedLine(na, start_pos, end_pos, (0, 255, 0), 1, tipLength = 0.03)
                    cv2.putText(na, p_str, text_pos, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = "{}/{}.png".format(output_folder, step_name)
        im = Image.fromarray(na)
        im.save(output_path)
        return im

    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
        step_name = self.vis_dataset_list[idx]
        img = self.read_img(self.vis_dataset[step_name]["Image_path"])
        predicates = self.vis_dataset[step_name]["Predicate_groundings"]
        bboxes = list(self.vis_dataset[step_name]["Bboxes"].values())
        types = list(self.vis_dataset[step_name]["Types"].values())
        detailed_types = list(self.vis_dataset[step_name]["Detailed_types"].values())
        priorities = list(self.vis_dataset[step_name]["Priorities"].values())
        direction_name_list = list(self.vis_dataset[step_name]["Directions"].values())
        direction_tensor_list = []
        for direction_name in direction_name_list:
            direction_tensor_list.append(self.direction_dict[direction_name])

        next_actions = list(self.vis_dataset[step_name]["Next_actions"].values())

        if self.gt_vis:
            output_folder = "gt_vis/{}".format(step_name.split('_')[0])
            self.gt_visualization(step_name, self.vis_dataset[step_name]["Image_path"], predicates, bboxes, detailed_types, output_folder)

        img = torch.Tensor(np.array(img))
        bboxes = torch.Tensor(np.array(bboxes))
        priorities = torch.Tensor(np.array(priorities))
        directions = torch.Tensor(np.array(direction_tensor_list))
        next_actions = torch.Tensor(np.array(next_actions)).to(torch.int64)
        next_actions = torch.nn.functional.one_hot(next_actions, num_classes=4).to(torch.float32)

        out_dict = {
            "step_name": step_name,
            "img": img,
            "predicates": predicates,
            "bboxes": bboxes,
            "types": types,
            "detailed_types": detailed_types,
            "priorities": priorities,
            "directions": directions,
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