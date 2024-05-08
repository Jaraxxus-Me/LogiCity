import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from logicity.utils.dataset import VisDataset


def CPU(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

def CUDA(x):
    if isinstance(x, list):
        return [sample.cuda() for sample in x]
    return x.cuda()

def compute_action_acc(pred, label):
    pred = CPU(pred)
    label = CPU(label)
    pred = np.argmax(pred, axis=-1)
    acc = np.sum(pred == label) / len(label)

    slow_correct_num = np.sum((pred==label)&(label==0))
    slow_gt_num = np.sum(label==0)
    normal_correct_num = np.sum((pred==label)&(label==1))
    normal_gt_num = np.sum(label==1)
    fast_correct_num = np.sum((pred==label)&(label==2))
    fast_gt_num = np.sum(label==2)
    stop_correct_num = np.sum((pred==label)&(label==3))
    stop_gt_num = np.sum(label==3)

    return acc, [slow_correct_num, slow_gt_num, normal_correct_num, normal_gt_num, \
            fast_correct_num, fast_gt_num, stop_correct_num, stop_gt_num]

def collate_fn(batch):
    batched_data = {}
    for key in batch[0].keys():
        if key == 'edge_index':
            batched_data[key] = [sample[key] for sample in batch]
        else:
            batched_data[key] = default_collate([item[key] for item in batch])

    return batched_data

def build_data_loader(data_config, test=False):
    vis_dataset_path = data_config['data_path']
    dataset_name = data_config['dataset_name']
    debug = data_config['debug']
    bs = data_config['batch_size']
    if not test:
        train_vis_dataset_path = os.path.join(vis_dataset_path, "train/train_{}.pkl".format(dataset_name))
        val_vis_dataset_path = os.path.join(vis_dataset_path, "val/val_{}.pkl".format(dataset_name))

        train_dataset = VisDataset(train_vis_dataset_path, debug=debug)
        train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn)
        val_dataset = VisDataset(val_vis_dataset_path)
        val_dataloader = DataLoader(val_dataset, batch_size=bs, collate_fn=collate_fn)
        return train_dataset, val_dataset, train_dataloader, val_dataloader
    else:
        test_vis_dataset_path = os.path.join(vis_dataset_path, "test/test_{}.pkl".format(dataset_name))
        test_dataset = VisDataset(test_vis_dataset_path)
        test_dataloader = DataLoader(test_dataset, batch_size=bs, collate_fn=collate_fn)
        return test_dataset, test_dataloader

def build_optimizer(params, opt_config):
    optim_type = opt_config['type']
    lr = opt_config['lr']
    lr_schedule = opt_config['scheduler']

    if optim_type == 'Adam':
        optimizer = torch.optim.Adam(params, lr=lr)
    elif optim_type == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=lr)
    else:
        raise ValueError("Optimizer type not supported.")

    return optimizer