import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import os
from logicity.utils.dataset import VisDataset
from torch.utils.data import DataLoader
from logicity.predictor import MODEL_BUILDER

def CPU(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

def CUDA(x):
    return x.cuda()

def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default='LogicityVisPredictorNLM', help="model name")
    parser.add_argument("--data_path", type=str, default='vis_dataset/easy_200')
    parser.add_argument("--mode", type=str, default='easy')
    parser.add_argument("--dataset_name", type=str, default='easy_200')
    parser.add_argument("--checkpoint_path", type=str, default='best.pth')
    parser.add_argument("--add_concept_loss", action='store_true', help='Add concept_loss in addition to action_loss.')
    return parser.parse_args()

    
def compute_action_acc(pred, label, types):
    pred = CPU(pred)
    label = CPU(label)
    pred = np.argmax(pred, axis=-1)
    label = np.argmax(label, axis=-1)
    acc = np.sum(pred == label) / len(label)
    
    is_car_mask = []
    for type in types:
        is_car_mask.append((type[0] == "Car"))
    car_correct_num = np.sum((pred == label)&is_car_mask)
    car_gt_num = np.sum(is_car_mask)
    car_acc =  car_correct_num / car_gt_num

    slow_correct_num = np.sum((pred==label)&(label==0))
    slow_gt_num = np.sum(label==0)
    normal_correct_num = np.sum((pred==label)&(label==1))
    normal_gt_num = np.sum(label==1)
    fast_correct_num = np.sum((pred==label)&(label==2))
    fast_gt_num = np.sum(label==2)
    stop_correct_num = np.sum((pred==label)&(label==3))
    stop_gt_num = np.sum(label==3)

    car_slow_correct_num = np.sum((pred==label)&(label==0)&is_car_mask)
    car_slow_gt_num = np.sum((label==0)&is_car_mask)
    car_normal_correct_num = np.sum((pred==label)&(label==1)&is_car_mask)
    car_normal_gt_num = np.sum((label==1)&is_car_mask)
    car_fast_correct_num = np.sum((pred==label)&(label==2)&is_car_mask)
    car_fast_gt_num = np.sum((label==2)&is_car_mask)
    car_stop_correct_num = np.sum((pred==label)&(label==3)&is_car_mask)
    car_stop_gt_num = np.sum((label==3)&is_car_mask)

    return acc, [slow_correct_num, slow_gt_num, normal_correct_num, normal_gt_num, \
            fast_correct_num, fast_gt_num, stop_correct_num, stop_gt_num], \
            car_acc, [car_correct_num, car_gt_num], [car_slow_correct_num, car_slow_gt_num, car_normal_correct_num, car_normal_gt_num, \
            car_fast_correct_num, car_fast_gt_num, car_stop_correct_num, car_stop_gt_num],


if __name__ == "__main__":
    args = get_parser()
    vis_dataset_path = args.data_path
    dataset_name = args.dataset_name
    mode = args.mode

    model = MODEL_BUILDER[args.model](mode)
    state_dict = torch.load(args.checkpoint_path)["model_state_dict"]
    model.load_state_dict(state_dict)
    model = CUDA(model)

    # prepare test set
    test_vis_dataset_path = os.path.join(vis_dataset_path, "test/test_{}.pkl".format(dataset_name))
    test_dataset = VisDataset(test_vis_dataset_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    loss_ce = nn.CrossEntropyLoss()
    loss_bce = nn.BCELoss()

    # evaluate the accuracy and loss on test set
    loss_test = 0.
    acc_test = 0.
    action_total = [0, 0, 0, 0, 0, 0, 0, 0]
    car_acc_test = 0.
    car_total = [0, 0]
    car_action_total = [0, 0, 0, 0, 0, 0, 0, 0]
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
            acc, action_results_list, car_acc, car_results_list, car_action_results_list = compute_action_acc(pred_actions, gt_actions, gt_types)
            acc_test += acc
            car_acc_test += car_acc
            for i, a in enumerate(action_results_list):
                action_total[i] += a
            for i, a in enumerate(car_results_list):
                car_total[i] += a
            for i, a in enumerate(car_action_results_list):
                car_action_total[i] += a

    loss_test /= len(test_dataset)
    acc_test /= len(test_dataset)
    car_acc_test /= len(test_dataset)

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



    car_slow_acc = car_action_total[0] / car_action_total[1]
    car_normal_acc = car_action_total[2] / car_action_total[3]
    car_fast_acc = car_action_total[4] / car_action_total[5]
    car_stop_acc = car_action_total[6] / car_action_total[7]
    car_action_avg_acc = (car_slow_acc + car_normal_acc + car_fast_acc + car_stop_acc) / 4
    car_action_weighted_acc = (car_action_total[0] + car_action_total[2] + car_action_total[4] + car_action_total[6]) / \
                            (car_action_total[1] + car_action_total[3] + car_action_total[5] + car_action_total[7])
    car_action_avg_acc_no_normal = (car_slow_acc + car_fast_acc + car_stop_acc) / 3
    car_action_weighted_acc_no_normal = (car_action_total[0] + car_action_total[4] + car_action_total[6]) / \
                            (car_action_total[1] + car_action_total[5] + car_action_total[7])
    print("Car Slow: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(car_action_total[0], car_action_total[1], car_slow_acc))
    print("Car Normal: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(car_action_total[2], car_action_total[3], car_normal_acc))
    print("Car Fast: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(car_action_total[4], car_action_total[5], car_fast_acc))
    print("Car Stop: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(car_action_total[6], car_action_total[7], car_stop_acc))
    print("Car Sample Avg Acc: {:.4f}".format(car_acc_test))
    print("Car: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(car_total[0], car_total[1], car_total[0]/car_total[1]))
    print("Car Action Avg Acc: {:.4f}".format(car_action_avg_acc))
    print("Car Action Weighted Acc: {:.4f}".format(car_action_weighted_acc))
    print("Car Action Avg Acc w/o Normal: {:.4f}".format(car_action_avg_acc_no_normal))
    print("Car Action Weighted Acc w/o Normal: {:.4f}".format(car_action_weighted_acc_no_normal))