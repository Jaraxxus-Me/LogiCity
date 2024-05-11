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
from logicity.utils.vis_utils import CPU, CUDA, build_data_loader, compute_action_acc

def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default='config/tasks/Vis/ResNetNLM/easy_200_fixed_modular.yaml', help='Path to the config file.')
    parser.add_argument("--exp", type=str, default='resnet_gnn')
    parser.add_argument('--only_supervise_car', default=True, help='Only supervise the car actions.')
    parser.add_argument("--ckpt", type=str, required=True, help='Path to the checkpoint file.')
    parser.add_argument("--add_concept_loss", default=True, help='Add concept_loss in addition to action_loss.')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_parser()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    data_config = config['Data']

    test_dataset, test_dataloader = build_data_loader(data_config, test=True)
    # train_dataset, test_dataset, train_dataloader, test_dataloader = build_data_loader(data_config)

    model_config = config['Model']
    model = MODEL_BUILDER[model_config['name']](model_config, config['Data']['mode'])
    state_dict = torch.load(args.ckpt)["model_state_dict"]
    model.load_state_dict(state_dict)
    model = CUDA(model)

    # evaluate the accuracy and loss on test set
    acc_test = 0.
    # unary_acc_test = 0.
    # binary_acc_test = 0.
    action_total = [0, 0, 0, 0, 0, 0, 0, 0]
    # correct_unary_total = [0, 0, 0, 0, 0, 0, 0, 0]
    # gt_unary_total = [0, 0, 0, 0, 0, 0, 0, 0]
    # correct_binary_total = [0, 0, 0, 0]
    # gt_binary_total = [0, 0, 0, 0]
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            gt_actions = CUDA(batch["next_actions"])
            pred_unary_concepts, pred_binary_concepts = model.grounding(CUDA(batch["img"]), CUDA(batch["bboxes"]), \
                                                                        CUDA(batch["directions"]), CUDA(batch["priorities"]), CUDA(batch["edge_index"]))
            pred_round_unary_concepts = (pred_unary_concepts > 0.5).float()
            pred_round_binary_concepts = (pred_binary_concepts > 0.5).float()

            if "GNN" in args.config:
                sliced_edge_concepts = model.create_sliced_edge_concepts(pred_round_binary_concepts, CUDA(batch["edge_index"]))
                batch_graph = model.create_batch_graph(pred_round_unary_concepts, sliced_edge_concepts, CUDA(batch["edge_index"])) 
                pred_actions = model.reasoning(batch_graph)
            else:
                assert "NLM" in args.config, "Only GNN and NLM are supported."
                feed_dict = {
                        "n": torch.tensor([pred_round_unary_concepts.shape[1]]*pred_round_unary_concepts.shape[0]), # [N] * (B*N)
                        "states": pred_round_unary_concepts, # BN x N x C_node
                        "relations": pred_round_binary_concepts, # BN x N x N x (C_edge+1)
                    }
                pred_actions = model.reasoning(feed_dict)

            if args.only_supervise_car:
                is_car_mask = CUDA(batch["car_mask"])
                is_car_mask = is_car_mask.bool().reshape(-1)
                pred_actions = pred_actions.reshape(-1, 4)[is_car_mask]
                gt_actions = gt_actions.reshape(-1)[is_car_mask]
            acc, action_results_list = compute_action_acc(pred_actions, gt_actions)
            acc_test += acc
            for i, a in enumerate(action_results_list):
                action_total[i] += a

            # unary_acc, unary_list = compute_concept_acc(pred_unary_concepts, gt_unary_concepts)
            # binary_acc, binary_list = compute_concept_acc(pred_binary_concepts, gt_binary_concepts)
            # unary_acc_test += unary_acc
            # binary_acc_test += binary_acc
            
            # for i, a in enumerate(unary_list[0]):
            #     correct_unary_total[i] += a
            # for i, a in enumerate(binary_list[0]):
            #     correct_binary_total[i] += a
            # for i, a in enumerate(unary_list[1]):
            #     gt_unary_total[i] += a
            # for i, a in enumerate(binary_list[1]):
            #     gt_binary_total[i] += a

    acc_test /= len(test_dataset)
    # unary_acc_test /= len(test_dataset)
    # binary_acc_test /= len(test_dataset)

    slow_acc = action_total[0] / action_total[1]
    normal_acc = action_total[2] / action_total[3]
    fast_acc = action_total[4] / action_total[5]
    stop_acc = action_total[6] / action_total[7]

    acc_total = [slow_acc, normal_acc, fast_acc, stop_acc]
    action_factor = 0
    action_weighted_acc = 0
    # filter unseen action
    for i in range(4):
        if action_total[2*i+1] == 0:
            print("Action {} is unseen.".format(i))
            continue
        action_factor += 1 / action_total[2*i+1]
        action_weighted_acc += acc_total[i] / action_total[2*i+1]
    action_weighted_acc /= action_factor
    
    # unary_factor = 0
    # for i in range(len(unary_list[0])):
    #     unary_factor += 1/gt_unary_total[i]
    # print("unary_factor", unary_factor)
    # unary_weighted_acc = 0
    # for i in range(len(unary_list[0])):
    #     print(f"correct_unary_total[{i}]: {correct_unary_total[i]}, gt_unary_total[{i}]: {gt_unary_total[i]}")
    #     unary_weighted_acc += (correct_unary_total[i]/gt_unary_total[i])/gt_unary_total[i]
    # unary_weighted_acc /= unary_factor

    # binary_factor = 0
    # for i in range(len(binary_list[0])):
    #     binary_factor += 1/gt_binary_total[i]
    # print("binary_factor", binary_factor)
    # binary_weighted_acc = 0
    # for i in range(len(binary_list[0])):
    #     print(f"correct_binary_total[{i}]: {correct_binary_total[i]}, gt_binary_total[{i}]: {gt_binary_total[i]}")
    #     binary_weighted_acc += (correct_binary_total[i]/gt_binary_total[i])/gt_binary_total[i]
    # binary_weighted_acc /= binary_factor

    print("Slow: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(action_total[0], action_total[1], slow_acc))
    print("Normal: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(action_total[2], action_total[3], normal_acc))
    print("Fast: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(action_total[4], action_total[5], fast_acc))
    print("Stop: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(action_total[6], action_total[7], stop_acc))
    print("Testing Sample Avg Acc: {:.4f}".format(acc_test))
    print("Action Weighted Acc: {:.4f}".format(action_weighted_acc))
    # print("Unary Concept Sample Avg Acc: {:.4f}".format(unary_acc_test))
    # print("Binary Concept Sample Avg Acc: {:.4f}".format(binary_acc_test))
    # print("Unary Concept Weighted Acc: {:.4f}".format(unary_weighted_acc))
    # print("Binary Concept Weighted Acc: {:.4f}".format(binary_weighted_acc))
    print()
