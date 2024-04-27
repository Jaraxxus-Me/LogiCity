import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import wandb
import yaml
import os
from logicity.utils.dataset import VisDataset
from logicity.utils.vis_utils import CPU, CUDA, build_data_loader, compute_action_acc, build_optimizer
from torch.utils.data import DataLoader
from logicity.predictor import MODEL_BUILDER

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = inputs.sigmoid()
        loss = -(1 - inputs) ** self.gamma * targets * torch.log(inputs) - inputs ** self.gamma * (1 - targets) * torch.log(1 - inputs)
        loss = loss.mean()
        return loss

def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default='config/tasks/Vis/ResNetGNN/easy_200_fixed_e2e.yaml', help='Path to the config file.')
    parser.add_argument("--exp", type=str, default='resnet_gnn_e2e')
    parser.add_argument("--modular", action='store_true', help='Train the model in a modular style.')
    parser.add_argument('--bilevel', action='store_true', help='Train the model in a bilevel style.')
    parser.add_argument('--only_supervise_car', default=True, help='Only supervise the car actions.')
    parser.add_argument('--add_concept_loss', default=True, help='Only supervise the car actions.')
    return parser.parse_args()

def train_modular(args):
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    data_config = config['Data']
    train_dataset, val_dataset, train_dataloader, val_dataloader = build_data_loader(data_config)

    model_config = config['Model']
    model = MODEL_BUILDER[model_config['name']](model_config, config['Data']['mode'])
    model = CUDA(model)

    grounding_opt_config = config["Optimizer"]["grounding"]
    grounding_params = list(model.node_concept_predictor.parameters()) + \
                        list(model.edge_predictor.parameters())
    grounding_optimizer = build_optimizer(grounding_params, grounding_opt_config)
    reasoning_opt_config = config["Optimizer"]["reasoning"]
    reasoning_optimizer = build_optimizer(model.reasoning_engine.parameters(), reasoning_opt_config)

    wandb.init(
        project = "logicity_vis_input",
        name = "{}_{}".format(args.exp, config['Data']['mode']),
        config = config,
    )

    loss_ce = nn.CrossEntropyLoss()
    loss_bce = nn.BCELoss()

    wandb.watch(model)
    best_acc = -1
    epochs = config['Optimizer']['epochs']
    for epoch in range(epochs):
        loss_train_concepts, loss_train_actions, loss_val = 0., 0., 0.
        acc_train, acc_val = 0., 0.

        action_total = [0, 0, 0, 0, 0, 0, 0, 0]

        iter_num = 0
        for batch in tqdm(train_dataloader):
            gt_actions = CUDA(batch["next_actions"])
            gt_unary_concepts = CUDA(batch["predicates"]["unary"])
            gt_binary_concepts = CUDA(batch["predicates"]["binary"])
            # TODO: for zhaoyu, gt_binary_concepts[:, -1] is the concept of "sees", use 11x10 matrix to represent the edge
            pred_unary_concepts, pred_binary_concepts = model.pred_concepts(CUDA(batch["img"]), CUDA(batch["bboxes"]), CUDA(batch["directions"]), CUDA(batch["priorities"]))
            # cut the gradient for nlm here
            pred_unary_concepts_nlm = pred_unary_concepts.detach()
            pred_binary_concepts_nlm = pred_binary_concepts.detach()

            pred_actions = model.reason(pred_unary_concepts_nlm, pred_binary_concepts_nlm)
            
            if args.only_supervise_car:
                is_car_mask = CUDA(batch["car_mask"])
                car_gt_num = torch.sum(is_car_mask)
                car_action_gt_num_list = []
                for i in range(4):
                    car_action_gt_num_list.append(torch.sum((gt_actions == i) * is_car_mask))
                car_action_gt_nums = torch.FloatTensor(car_action_gt_num_list).to(is_car_mask.device)
                class_weights = car_gt_num / (car_action_gt_nums + 1e-6)
                is_car_mask = is_car_mask.bool().reshape(-1)
                pred_actions = pred_actions.reshape(-1, 4)[is_car_mask]
                gt_actions = gt_actions.reshape(-1)[is_car_mask]
                loss_ce = nn.CrossEntropyLoss(weight=CUDA(class_weights))
                loss_actions = loss_ce(pred_actions, gt_actions)
            else:
                loss_actions = loss_ce(pred_actions.reshape(-1, 4), gt_actions.reshape(-1))
            
            if args.add_concept_loss:
                loss_concepts = loss_bce(pred_unary_concepts.reshape(-1), gt_unary_concepts.reshape(-1)) \
                                + loss_bce(pred_binary_concepts.reshape(-1), gt_binary_concepts.reshape(-1))
            
            grounding_optimizer.zero_grad()
            loss_concepts.backward()
            grounding_optimizer.step()

            reasoning_optimizer.zero_grad()
            loss_actions.backward()
            reasoning_optimizer.step()

            loss_train_concepts += loss_concepts.item()
            loss_train_actions += loss_actions.item()
            acc, action_results_list = compute_action_acc(pred_actions, gt_actions)
            acc_train += acc
            for i, a in enumerate(action_results_list):
                action_total[i] += a

            iter_num += 1
            # validation
            if iter_num % len(train_dataloader) == 0:
                # evaluate the accuracy and loss on val set
                with torch.no_grad():
                    val_action_total = [0, 0, 0, 0, 0, 0, 0, 0]
                    for batch in tqdm(val_dataloader):
                        gt_actions = CUDA(batch["next_actions"])
                        gt_unary_concepts = CUDA(batch["predicates"]["unary"])
                        gt_binary_concepts = CUDA(batch["predicates"]["binary"])
                        pred_actions, pred_unary_concepts, pred_binary_concepts = model(CUDA(batch["img"]), CUDA(batch["bboxes"]), CUDA(batch["directions"]), CUDA(batch["priorities"]))
                        
                        if args.only_supervise_car:
                            is_car_mask = CUDA(batch["car_mask"])
                            car_gt_num = torch.sum(is_car_mask)
                            car_action_gt_num_list = []
                            for i in range(4):
                                car_action_gt_num_list.append(torch.sum((gt_actions == i) * is_car_mask))
                            car_action_gt_nums = torch.FloatTensor(car_action_gt_num_list).to(is_car_mask.device)
                            class_weights = car_gt_num / (car_action_gt_nums + 1e-6)
                            is_car_mask = is_car_mask.bool().reshape(-1)
                            pred_actions = pred_actions.reshape(-1, 4)[is_car_mask]
                            gt_actions = gt_actions.reshape(-1)[is_car_mask]
                            loss_ce = nn.CrossEntropyLoss(weight=CUDA(class_weights))
                            loss_actions = loss_ce(pred_actions, gt_actions)
                        else:
                            loss_actions = loss_ce(pred_actions, gt_actions)
                        loss = loss_actions
                        loss_concepts = loss_bce(pred_unary_concepts, gt_unary_concepts) \
                                        + loss_bce(pred_binary_concepts, gt_binary_concepts)
                        loss += loss_concepts
                        loss_val += loss.item()
                        acc, val_action_results_list = compute_action_acc(pred_actions, gt_actions)
                        acc_val += acc

                        for i, a in enumerate(val_action_results_list):
                            val_action_total[i] += a

                loss_val /= len(val_dataset)
                acc_val /= len(val_dataset)
                print("Epoch: {}, Iter: {}, Validation Loss: {:.4f}, Sample Avg Acc: {:.4f}".format(
                    epoch, iter_num, loss_val, acc_val))
                
                val_slow_acc = val_action_total[0] / val_action_total[1]
                val_normal_acc = val_action_total[2] / val_action_total[3]
                val_fast_acc = val_action_total[4] / val_action_total[5]
                val_stop_acc = val_action_total[6] / val_action_total[7]

                val_action_factor = 1 / val_action_total[1] + 1 / val_action_total[3] \
                                + 1 / val_action_total[5] + 1 / val_action_total[7]

                val_action_weighted_acc = (val_slow_acc / val_action_total[1] \
                                           + val_normal_acc / val_action_total[3] \
                                           + val_fast_acc / val_action_total[5] \
                                           + val_stop_acc / val_action_total[7]) / val_action_factor

                print("Slow: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(val_action_total[0], val_action_total[1], val_slow_acc))
                print("Normal: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(val_action_total[2], val_action_total[3], val_normal_acc))
                print("Fast: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(val_action_total[4], val_action_total[5], val_fast_acc))
                print("Stop: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(val_action_total[6], val_action_total[7], val_stop_acc))
                print("Action Weighted Acc: {:.4f}".format(val_action_weighted_acc))
                
                wandb.log({
                    'iter': epoch*len(train_dataloader) + iter_num,
                    'loss_val': loss_val,
                    'acc_val': acc_val,
                    'acc_val_weighted': val_action_weighted_acc,
                })

        loss_train_concepts /= len(train_dataset)
        loss_train_actions /= len(train_dataset)
        acc_train /= len(train_dataset)
        print("Epoch: {}, Training Loss (Concepts): {:.4f}, Training Loss (Actions): {:.4f}, Sample Avg Acc: {:.4f}".format(
            epoch, loss_train_concepts, loss_train_actions, acc_train))
        slow_acc = action_total[0] / action_total[1]
        normal_acc = action_total[2] / action_total[3]
        fast_acc = action_total[4] / action_total[5]
        stop_acc = action_total[6] / action_total[7]

        action_factor = 1 / action_total[1] + 1 / action_total[3] \
                        + 1 / action_total[5] + 1 / action_total[7]

        action_weighted_acc = (slow_acc / action_total[1] \
                                    + normal_acc / action_total[3] \
                                    + fast_acc / action_total[5] \
                                    + stop_acc / action_total[7]) / action_factor

        print("Slow: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(action_total[0], action_total[1], slow_acc))
        print("Normal: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(action_total[2], action_total[3], normal_acc))
        print("Fast: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(action_total[4], action_total[5], fast_acc))
        print("Stop: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(action_total[6], action_total[7], stop_acc))
        print("Action Weighted Acc: {:.4f}".format(action_weighted_acc))

        wandb.log({
            'epoch': epoch,
            'learning rate (grounding)': grounding_optimizer.state_dict()['param_groups'][0]['lr'],
            'learning rate (reasoning)': reasoning_optimizer.state_dict()['param_groups'][0]['lr'],
            'loss_train_concept': loss_train_concepts,
            'loss_train_action': loss_train_actions,
            'acc_train': acc_train,
            'acc_train_weighted': action_weighted_acc,
        })

        if (epoch + 1) % 2 == 0:
            if not os.path.exists("vis_input_weights/{}/{}".format(config['Data']['mode'], args.exp)):
                os.makedirs("vis_input_weights/{}/{}".format(config['Data']['mode'], args.exp))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'grounder_optimizer_state_dict': grounding_optimizer.state_dict(),
                'reasoning_optimizer_state_dict': reasoning_optimizer.state_dict(),
                'loss': loss_val,
            }, "vis_input_weights/{}/{}/{}_epoch{}_valacc{:.4f}.pth".format(config['Data']['mode'], args.exp, args.exp, epoch, val_action_weighted_acc))
            if best_acc < val_action_weighted_acc:
                best_acc = val_action_weighted_acc
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'grounder_optimizer_state_dict': grounding_optimizer.state_dict(),
                'reasoning_optimizer_state_dict': reasoning_optimizer.state_dict(),
                'loss': loss_val,
            }, "vis_input_weights/{}/{}/{}_best.pth".format(config['Data']['mode'], args.exp, args.exp))

    wandb.finish()

def train_e2e(args):
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    data_config = config['Data']
    train_dataset, val_dataset, train_dataloader, val_dataloader = build_data_loader(data_config)

    model_config = config['Model']
    model = MODEL_BUILDER[model_config['name']](model_config, config['Data']['mode'])
    model = CUDA(model)

    assert "whole" in config["Optimizer"], "The whole model should be optimized e2e."
    whole_opt_config = config["Optimizer"]["whole"]
    whole_params = list(model.node_concept_predictor.parameters()) + \
                        list(model.edge_predictor.parameters()) + \
                        list(model.reasoning_engine.parameters())
    whole_optimizer = build_optimizer(whole_params, whole_opt_config)

    wandb.init(
        project = "logicity_vis_input",
        name = "{}_{}".format(args.exp, config['Data']['mode']),
        config = config,
    )

    loss_ce = nn.CrossEntropyLoss()
    loss_bce = nn.BCELoss()

    wandb.watch(model)
    best_acc = -1
    epochs = config['Optimizer']['epochs']
    for epoch in range(epochs):
        loss_train_concepts, loss_train_actions, loss_val = 0., 0., 0.
        acc_train, acc_val = 0., 0.

        action_total = [0, 0, 0, 0, 0, 0, 0, 0]

        iter_num = 0
        for batch in tqdm(train_dataloader):
            gt_actions = CUDA(batch["next_actions"])
            gt_unary_concepts = CUDA(batch["predicates"]["unary"])
            gt_binary_concepts = CUDA(batch["predicates"]["binary"])
            # TODO: for zhaoyu, gt_binary_concepts[:, -1] is the concept of "sees", use 11x10 matrix to represent the edge
            pred_unary_concepts, pred_binary_concepts = model.pred_concepts(CUDA(batch["img"]), CUDA(batch["bboxes"]), CUDA(batch["directions"]), CUDA(batch["priorities"]))
            pred_actions = model.reason(pred_unary_concepts, pred_binary_concepts)
            
            if args.only_supervise_car:
                is_car_mask = CUDA(batch["car_mask"])
                car_gt_num = torch.sum(is_car_mask)
                car_action_gt_num_list = []
                for i in range(4):
                    car_action_gt_num_list.append(torch.sum((gt_actions == i) * is_car_mask))
                car_action_gt_nums = torch.FloatTensor(car_action_gt_num_list).to(is_car_mask.device)
                class_weights = car_gt_num / (car_action_gt_nums + 1e-6)
                is_car_mask = is_car_mask.bool().reshape(-1)
                pred_actions = pred_actions.reshape(-1, 4)[is_car_mask]
                gt_actions = gt_actions.reshape(-1)[is_car_mask]
                loss_ce = nn.CrossEntropyLoss(weight=CUDA(class_weights))
                loss_actions = loss_ce(pred_actions, gt_actions)
            else:
                loss_actions = loss_ce(pred_actions.reshape(-1, 4), gt_actions.reshape(-1))
            
            # if args.add_concept_loss:
            #     loss_concepts = loss_bce(pred_unary_concepts.reshape(-1), gt_unary_concepts.reshape(-1)) \
            #                     + loss_bce(pred_binary_concepts.reshape(-1), gt_binary_concepts.reshape(-1))
            
            whole_optimizer.zero_grad()
            loss_actions.backward()
            whole_optimizer.step()

            loss_train_actions += loss_actions.item()
            acc, action_results_list = compute_action_acc(pred_actions, gt_actions)
            acc_train += acc
            for i, a in enumerate(action_results_list):
                action_total[i] += a

            iter_num += 1
            # validation
            if iter_num % len(train_dataloader) == 0:
                # evaluate the accuracy and loss on val set
                with torch.no_grad():
                    val_action_total = [0, 0, 0, 0, 0, 0, 0, 0]
                    for batch in tqdm(val_dataloader):
                        gt_actions = CUDA(batch["next_actions"])
                        gt_unary_concepts = CUDA(batch["predicates"]["unary"])
                        gt_binary_concepts = CUDA(batch["predicates"]["binary"])
                        pred_actions, pred_unary_concepts, pred_binary_concepts = model(CUDA(batch["img"]), CUDA(batch["bboxes"]), CUDA(batch["directions"]), CUDA(batch["priorities"]))
                        
                        if args.only_supervise_car:
                            is_car_mask = CUDA(batch["car_mask"])
                            car_gt_num = torch.sum(is_car_mask)
                            car_action_gt_num_list = []
                            for i in range(4):
                                car_action_gt_num_list.append(torch.sum((gt_actions == i) * is_car_mask))
                            car_action_gt_nums = torch.FloatTensor(car_action_gt_num_list).to(is_car_mask.device)
                            class_weights = car_gt_num / (car_action_gt_nums + 1e-6)
                            is_car_mask = is_car_mask.bool().reshape(-1)
                            pred_actions = pred_actions.reshape(-1, 4)[is_car_mask]
                            gt_actions = gt_actions.reshape(-1)[is_car_mask]
                            loss_ce = nn.CrossEntropyLoss(weight=CUDA(class_weights))
                            loss_actions = loss_ce(pred_actions, gt_actions)
                        else:
                            loss_actions = loss_ce(pred_actions, gt_actions)
                        loss = loss_actions
                        # loss_concepts = loss_bce(pred_unary_concepts, gt_unary_concepts) \
                        #                 + loss_bce(pred_binary_concepts, gt_binary_concepts)
                        # loss += loss_concepts
                        loss_val += loss.item()
                        acc, val_action_results_list = compute_action_acc(pred_actions, gt_actions)
                        acc_val += acc

                        for i, a in enumerate(val_action_results_list):
                            val_action_total[i] += a

                loss_val /= len(val_dataset)
                acc_val /= len(val_dataset)
                
                val_slow_acc = val_action_total[0] / val_action_total[1]
                val_normal_acc = val_action_total[2] / val_action_total[3]
                val_fast_acc = val_action_total[4] / val_action_total[5]
                val_stop_acc = val_action_total[6] / val_action_total[7]

                val_action_factor = 1 / val_action_total[1] + 1 / val_action_total[3] \
                                + 1 / val_action_total[5] + 1 / val_action_total[7]

                val_action_weighted_acc = (val_slow_acc / val_action_total[1] \
                                           + val_normal_acc / val_action_total[3] \
                                           + val_fast_acc / val_action_total[5] \
                                           + val_stop_acc / val_action_total[7]) / val_action_factor
                print("Epoch: {}, Iter: {}, Validation Loss: {:.4f}, Sample Avg Acc: {:.4f}".format(
                    epoch, iter_num, loss_val, acc_val))
                print("Slow: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(val_action_total[0], val_action_total[1], val_slow_acc))
                print("Normal: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(val_action_total[2], val_action_total[3], val_normal_acc))
                print("Fast: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(val_action_total[4], val_action_total[5], val_fast_acc))
                print("Stop: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(val_action_total[6], val_action_total[7], val_stop_acc))
                print("Action Weighted Acc: {:.4f}".format(val_action_weighted_acc))
                
                wandb.log({
                    'iter': epoch*len(train_dataloader) + iter_num,
                    'loss_val': loss_val,
                    'acc_val': acc_val,
                    'acc_val_weighted': val_action_weighted_acc,
                })

        loss_train_concepts /= len(train_dataset)
        loss_train_actions /= len(train_dataset)
        acc_train /= len(train_dataset)
        slow_acc = action_total[0] / action_total[1]
        normal_acc = action_total[2] / action_total[3]
        fast_acc = action_total[4] / action_total[5]
        stop_acc = action_total[6] / action_total[7]

        action_factor = 1 / action_total[1] + 1 / action_total[3] \
                        + 1 / action_total[5] + 1 / action_total[7]

        action_weighted_acc = (slow_acc / action_total[1] \
                                    + normal_acc / action_total[3] \
                                    + fast_acc / action_total[5] \
                                    + stop_acc / action_total[7]) / action_factor
        print("Epoch: {}, Training Loss (Concepts): {:.4f}, Training Loss (Actions): {:.4f}, Sample Avg Acc: {:.4f}".format(
            epoch, loss_train_concepts, loss_train_actions, acc_train))
        print("Slow: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(action_total[0], action_total[1], slow_acc))
        print("Normal: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(action_total[2], action_total[3], normal_acc))
        print("Fast: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(action_total[4], action_total[5], fast_acc))
        print("Stop: Correct_num: {}, Total_num: {}, Acc: {:.4f}".format(action_total[6], action_total[7], stop_acc))
        print("Action Weighted Acc: {:.4f}".format(action_weighted_acc))

        wandb.log({
            'epoch': epoch,
            'learning rate': whole_optimizer.state_dict()['param_groups'][0]['lr'],
            'loss_train_concept': loss_train_concepts,
            'loss_train_action': loss_train_actions,
            'acc_train': acc_train,
            'acc_train_weighted': action_weighted_acc,
        })

        if (epoch + 1) % 2 == 0:
            if not os.path.exists("vis_input_weights/{}/{}".format(config['Data']['mode'], args.exp)):
                os.makedirs("vis_input_weights/{}/{}".format(config['Data']['mode'], args.exp))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': whole_optimizer.state_dict(),
                'loss': loss_val,
            }, "vis_input_weights/{}/{}/{}_epoch{}_valacc{:.4f}.pth".format(config['Data']['mode'], args.exp, args.exp, epoch, val_action_weighted_acc))
            if best_acc < val_action_weighted_acc:
                best_acc = val_action_weighted_acc
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': whole_optimizer.state_dict(),
                'loss': loss_val,
            }, "vis_input_weights/{}/{}/{}_best.pth".format(config['Data']['mode'], args.exp, args.exp))

    wandb.finish()

def train_bilevel(args):
    vis_dataset_path = args.data_path
    mode = args.mode
    dataset_name = args.dataset_name
    lr = args.lr
    epochs = args.epochs

    config = {
        "dataset": dataset_name,
        "epochs": epochs,
        "learning_rate": lr,
    }
    print(config)

    wandb.init(
        project = "logicity_vis_input",
        name = "{}_{}".format(args.exp, mode),
        config = config,
    )

    model = MODEL_BUILDER[args.model](mode)
    model = CUDA(model)

    outer_optimizer = torch.optim.Adam(model.perceptor.parameters(), lr=lr)
    inner_optimizer = torch.optim.Adam(model.reasoning_engine.parameters(), lr=lr)

    # prepare train and val set
    train_vis_dataset_path = os.path.join(vis_dataset_path, "train/train_{}.pkl".format(dataset_name))
    val_vis_dataset_path = os.path.join(vis_dataset_path, "val/val_{}.pkl".format(dataset_name))
    train_dataset = VisDataset(train_vis_dataset_path, debug=True)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    # val_dataset = VisDataset(val_vis_dataset_path)
    # val_dataloader = DataLoader(val_dataset, batch_size=1)
    val_dataset = train_dataset
    val_dataloader = DataLoader(train_dataset, batch_size=1)

    loss_ce = nn.CrossEntropyLoss()
    loss_bce = nn.BCELoss()

    wandb.watch(model)
    best_acc = -1
    train_target = ['inner', 'outer'] * (epochs // 2)
    for epoch in range(epochs):
        action_loss_train, concept_loss_train, action_loss_val, concept_loss_val = 0., 0., 0., 0.
        acc_train, acc_val = 0., 0.

        for batch in tqdm(train_dataloader):
            gt_actions = CUDA(batch["next_actions"][0])
            gt_unary_concepts = CUDA(batch["predicates"]["unary"][0])
            gt_binary_concepts = CUDA(batch["predicates"]["binary"][0])
            pred_actions, pred_unary_concepts, pred_binary_concepts = model(CUDA(batch["img"]), CUDA(batch["bboxes"]), CUDA(batch["directions"]), CUDA(batch["priorities"]))
            if train_target[epoch] == 'inner':
                loss_concepts = loss_bce(pred_unary_concepts, gt_unary_concepts) \
                                + loss_bce(pred_binary_concepts, gt_binary_concepts)
                loss_concepts.backward()
                action_loss_train += loss_actions.item()
                inner_optimizer.step()
                inner_optimizer.zero_grad()
                cur_lr = inner_optimizer.state_dict()['param_groups'][0]['lr']
            elif train_target[epoch] == 'outer':
                is_car_mask = CUDA(batch["car_mask"])
                car_gt_num = torch.sum(is_car_mask)
                car_action_gt_num_list = []
                for i in range(4):
                    car_action_gt_num_list.append(torch.sum((gt_actions == i) * is_car_mask))
                car_action_gt_nums = torch.FloatTensor(car_action_gt_num_list).to(is_car_mask.device)
                class_weights = car_gt_num / (car_action_gt_nums + 1e-6)
                is_car_mask = is_car_mask.bool().reshape(-1)
                pred_actions = pred_actions.reshape(-1, 4)[is_car_mask]
                gt_actions = gt_actions.reshape(-1)[is_car_mask]
                loss_ce = nn.CrossEntropyLoss(weight=CUDA(class_weights))
                loss_actions = loss_ce(pred_actions, gt_actions)
                loss_actions.backward()
                concept_loss_train += loss_concepts.item()
                outer_optimizer.step()
                outer_optimizer.zero_grad()
                cur_lr = outer_optimizer.state_dict()['param_groups'][0]['lr']

            acc = compute_action_acc(pred_actions, gt_actions)
            acc_train += acc

        action_loss_train /= len(train_dataset)
        concept_loss_train /= len(train_dataset)
        acc_train /= len(train_dataset)
        print("Epoch: {}, Training Action Loss: {:.4f}, Concept Loss: {:.4f}, Acc: {:.4f}, lr: {}".format(
            epoch, action_loss_train, concept_loss_train, acc_train, cur_lr))

        # evaluate the accuracy and loss on val set
        # with torch.no_grad():
        #     for batch in tqdm(val_dataloader):
        #         gt_actions = CUDA(batch["next_actions"][0])
        #         gt_unary_concepts = CUDA(batch["predicates"]["unary"][0])
        #         gt_binary_concepts = CUDA(batch["predicates"]["binary"][0])
        #         pred_actions, pred_unary_concepts, pred_binary_concepts = model(CUDA(batch["img"]), CUDA(batch["bboxes"]), CUDA(batch["directions"]), CUDA(batch["priorities"]))
        #         loss_actions = loss_ce(pred_actions, gt_actions)
        #         loss_concepts = loss_bce(pred_unary_concepts, gt_unary_concepts) \
        #                         + loss_bce(pred_binary_concepts, gt_binary_concepts)
        #         action_loss_val += loss_actions.item()
        #         concept_loss_val += loss_concepts.item()
        #         acc = compute_action_acc(pred_actions, gt_actions)
        #         acc_val += acc
        
        # action_loss_val /= len(val_dataset)
        # concept_loss_val /= len(val_dataset)
        # acc_val /= len(val_dataset)
        # print("Epoch: {}, Validation Action Loss: {:.4f}, Concept Loss: {:.4f}, Acc: {:.4f}".format(
        #     epoch, action_loss_val, concept_loss_val, acc_val))

        # wandb.log({
        #     'epoch': epoch,
        #     'learning rate': cur_lr,
        #     'action_loss_train': action_loss_train,
        #     'concept_loss_train': concept_loss_train,
        #     'action_loss_val': action_loss_val,
        #     'concept_loss_val': concept_loss_val,
        #     'acc_train': acc_train,
        #     'acc_val': acc_val,
        # })

        # if (epoch + 1) % 5 == 0:
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'inner_optimizer_state_dict': inner_optimizer.state_dict(),
        #         'outer_optimizer_state_dict': outer_optimizer.state_dict(),
        #         'action_loss_val': action_loss_val,
        #         'concept_loss_val': concept_loss_val,
        #     }, "vis_input_weights/{}/{}_lr{}_epoch{}_valacc{:.4f}.pth".format(mode, args.exp, lr, epoch, acc_val))
        #     if best_acc < acc_val:
        #         best_acc = acc_val
        #         torch.save({
        #             'epoch': epoch,
        #             'model_state_dict': model.state_dict(),
        #             'inner_optimizer_state_dict': inner_optimizer.state_dict(),
        #             'outer_optimizer_state_dict': outer_optimizer.state_dict(),
        #             'action_loss_val': action_loss_val,
        #             'concept_loss_val': concept_loss_val,
        #         }, "vis_input_weights/{}/{}_best.pth".format(mode, args.exp))

    wandb.finish()

if __name__ == "__main__":
    args = get_parser()
    if args.bilevel:
        train_bilevel(args)
    else:
        if args.modular:
            train_modular(args)
        else:
            train_e2e(args)