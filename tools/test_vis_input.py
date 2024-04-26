import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import os
import yaml
from logicity.utils.dataset import VisDataset
from torch.utils.data import DataLoader
from logicity.predictor import MODEL_BUILDER
from logicity.utils.vis_utils import CUDA, build_data_loader, compute_action_acc

def CPU(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

def CUDA(x):
    return x.cuda()

def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default='config/tasks/Vis/ResNetNLM/easy_200_fixed.yaml', help='Path to the config file.')
    parser.add_argument("--exp", type=str, default='resnet_nlm')
    parser.add_argument('--only_supervise_car', default=True, help='Only supervise the car actions.')
    parser.add_argument("--ckpt", type=str, required=True, help='Path to the checkpoint file.')
    parser.add_argument("--add_concept_loss", default=True, help='Add concept_loss in addition to action_loss.')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_parser()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    data_config = config['Data']

    test_dataset, test_dataloader = build_data_loader(data_config, test=True)

    model_config = config['Model']
    model = MODEL_BUILDER[model_config['name']](model_config, config['Data']['mode'])
    model = CUDA(model)

    loss_ce = nn.CrossEntropyLoss()
    loss_bce = nn.BCELoss()

    # evaluate the accuracy and loss on test set
    loss_test = 0.
    acc_test = 0.
    action_total = [0, 0, 0, 0, 0, 0, 0, 0]
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            gt_types = batch["types"]
            gt_actions = CUDA(batch["next_actions"][0])
            gt_unary_concepts = CUDA(batch["predicates"]["unary"][0])
            gt_binary_concepts = CUDA(batch["predicates"]["binary"][0])
            pred_actions, pred_unary_concepts, pred_binary_concepts = model(CUDA(batch["img"]), CUDA(batch["bboxes"]), CUDA(batch["directions"]), CUDA(batch["priorities"]))
            loss_actions = loss_ce(pred_actions, gt_actions)
            loss = loss_actions
            if args.add_concept_loss:
                loss_concepts = loss_bce(pred_unary_concepts, gt_unary_concepts) \
                                + loss_bce(pred_binary_concepts, gt_binary_concepts)
                loss += loss_concepts            
            loss_test += loss.item()
            acc, action_results_list = compute_action_acc(pred_actions, gt_actions)
            acc_test += acc
            for i, a in enumerate(action_results_list):
                action_total[i] += a

    loss_test /= len(test_dataset)
    acc_test /= len(test_dataset)

    slow_acc = action_total[0] / action_total[1]
    normal_acc = action_total[2] / action_total[3]
    fast_acc = action_total[4] / action_total[5]
    stop_acc = action_total[6] / action_total[7]
    action_avg_acc = (slow_acc + normal_acc + fast_acc + stop_acc) / 4
    action_weighted_acc = (action_total[0] + action_total[2] + action_total[4] + action_total[6]) / \
                            (action_total[1] + action_total[3] + action_total[5] + action_total[7])
    action_avg_acc_no_normal = (slow_acc + fast_acc + stop_acc) / 3
    action_weighted_acc_no_normal = (action_total[0] + action_total[4] + action_total[6]) / \
                            (action_total[1] + action_total[5] + action_total[7])
    print("Slow: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(action_total[0], action_total[1], slow_acc))
    print("Normal: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(action_total[2], action_total[3], normal_acc))
    print("Fast: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(action_total[4], action_total[5], fast_acc))
    print("Stop: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(action_total[6], action_total[7], stop_acc))
    print("Testing Loss: {:.4f}, Sample Avg Acc: {:.4f}".format(loss_test, acc_test))
    print("Action Avg Acc: {:.4f}".format(action_avg_acc))
    print("Action Weighted Acc: {:.4f}".format(action_weighted_acc))
    print("Action Avg Acc w/o Normal: {:.4f}".format(action_avg_acc_no_normal))
    print("Action Weighted Acc w/o Normal: {:.4f}".format(action_weighted_acc_no_normal))