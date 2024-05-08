import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
# import wandb
import yaml
import os
from logicity.utils.vis_utils import CPU, CUDA, build_data_loader, compute_action_acc, build_optimizer
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
    parser.add_argument("--config", type=str, default='config/tasks/Vis/ResNetNLM/easy_200_fixed_modular.yaml', help='Path to the config file.')
    parser.add_argument("--exp", type=str, default='resnet_gnn_e2e')
    parser.add_argument("--modular", action='store_true', help='Train the model in a modular style.')
    parser.add_argument('--only_supervise_car', default=True, help='Only supervise the car actions.')
    parser.add_argument('--add_concept_loss', default=True, help='Only supervise the car actions.')
    return parser.parse_args()


def get_gt_binary_concepts(batch_relation_matrix, batch_edge_index):
    gt_binary_concepts = []

    for b in range(len(batch_relation_matrix)):
        relation_matrix = batch_relation_matrix[b]
        src_nodes, dst_nodes = batch_edge_index[b]
        for src, dst in zip(src_nodes, dst_nodes):
            gt_binary_concepts.append(relation_matrix[dst, src])

    gt_binary_concepts_tensor = torch.stack(gt_binary_concepts, dim=0)

    return gt_binary_concepts_tensor


def train_modular(args):
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    data_config = config['Data']
    train_dataset, val_dataset, train_dataloader, val_dataloader = build_data_loader(data_config)

    model_config = config['Model']
    model = MODEL_BUILDER[model_config['name']](model_config, config['Data']['mode'])
    model = CUDA(model)

    grounding_opt_config = config["Optimizer"]["grounding"]
    grounding_optimizer = build_optimizer(model.grounding_net.parameters(), grounding_opt_config)

    reasoning_opt_config = config["Optimizer"]["reasoning"]
    reasoning_optimizer = build_optimizer(model.reasoning_net.parameters(), reasoning_opt_config)

    # wandb.init(
    #     project = "logicity_vis_input",
    #     name = "{}_{}".format(args.exp, config['Data']['mode']),
    #     config = config,
    # )

    loss_ce = nn.CrossEntropyLoss()
    loss_bce = nn.BCELoss()

    # wandb.watch(model)
    best_acc = -1
    epochs = config['Optimizer']['epochs']
    for epoch in range(epochs):
        loss_train_concepts, loss_train_actions, loss_val = 0., 0., 0.
        acc_train, acc_val = 0., 0.

        action_total = [0, 0, 0, 0, 0, 0, 0, 0]

        iter_num = 0
        model.train()
        for batch in tqdm(train_dataloader):
            gt_actions = CUDA(batch["next_actions"])
            gt_unary_concepts = CUDA(batch["predicates"]["unary"]) # B x N x 8
            gt_binary_concepts = CUDA(get_gt_binary_concepts(batch["predicates"]["binary"], batch["edge_index"])) # |E| x 3

            pred_unary_concepts, pred_binary_concepts = model.grounding(CUDA(batch["img"]), CUDA(batch["bboxes"]), CUDA(batch["directions"]), CUDA(batch["priorities"]), CUDA(batch["edge_index"]))
            sliced_edge_concepts = model.create_sliced_edge_concepts(gt_binary_concepts, CUDA(batch["edge_index"]))
            batch_graph = model.create_batch_graph(gt_unary_concepts, sliced_edge_concepts, CUDA(batch["edge_index"])) 
            pred_actions = model.reasoning(batch_graph)
            
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
            model.eval()
            if iter_num % len(train_dataloader) == 0 and (not data_config["debug"]):    
                # evaluate the accuracy and loss on val set
                with torch.no_grad():
                    val_action_total = [0, 0, 0, 0, 0, 0, 0, 0]
                    for batch in tqdm(val_dataloader):
                        gt_actions = CUDA(batch["next_actions"])
                        gt_unary_concepts = CUDA(batch["predicates"]["unary"]) # B x N x 8
                        gt_binary_concepts = CUDA(get_gt_binary_concepts(batch["predicates"]["binary"], batch["edge_index"])) # |E| x 3

                        pred_unary_concepts, pred_binary_concepts = model.grounding(CUDA(batch["img"]), CUDA(batch["bboxes"]), CUDA(batch["directions"]), CUDA(batch["priorities"]), CUDA(batch["edge_index"]))
                        
                        pred_round_unary_concepts = (pred_unary_concepts > 0.5).float()
                        pred_round_binary_concepts = (pred_binary_concepts > 0.5).float()

                        sliced_edge_concepts = model.create_sliced_edge_concepts(pred_round_binary_concepts, CUDA(batch["edge_index"]))
                        batch_graph = model.create_batch_graph(pred_round_unary_concepts, sliced_edge_concepts, CUDA(batch["edge_index"])) 
                        pred_actions = model.reasoning(batch_graph)
                        
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
                
                print({
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

        print({
            'epoch': epoch,
            'learning rate (grounding)': grounding_optimizer.state_dict()['param_groups'][0]['lr'],
            'learning rate (reasoning)': reasoning_optimizer.state_dict()['param_groups'][0]['lr'],
            'loss_train_concept': loss_train_concepts,
            'loss_train_action': loss_train_actions,
            'acc_train': acc_train,
            'acc_train_weighted': action_weighted_acc,
        })

        if (epoch + 1) % 2 == 0 and (not data_config["debug"]):    
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

    # wandb.finish()

def train_e2e(args):
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    data_config = config['Data']
    train_dataset, val_dataset, train_dataloader, val_dataloader = build_data_loader(data_config)

    model_config = config['Model']
    model = MODEL_BUILDER[model_config['name']](model_config, config['Data']['mode'])
    model = CUDA(model)

    assert "whole" in config["Optimizer"], "The whole model should be optimized e2e."
    whole_opt_config = config["Optimizer"]["whole"]
    parameters = model.parameters()
    optimizer = build_optimizer(parameters, whole_opt_config)

    # wandb.init(
    #     project = "logicity_vis_input",
    #     name = "{}_{}".format(args.exp, config['Data']['mode']),
    #     config = config,
    # )

    loss_ce = nn.CrossEntropyLoss()
    # wandb.watch(model)
    best_acc = -1
    epochs = config['Optimizer']['epochs']
    for epoch in range(epochs):
        loss_train_concepts, loss_train_actions, loss_val = 0., 0., 0.
        acc_train, acc_val = 0., 0.

        action_total = [0, 0, 0, 0, 0, 0, 0, 0]

        iter_num = 0
        model.train()
        for batch in tqdm(train_dataloader):
            gt_actions = CUDA(batch["next_actions"])
            pred_actions = model(CUDA(batch["img"]), CUDA(batch["bboxes"]), CUDA(batch["directions"]), CUDA(batch["priorities"]), CUDA(batch["edge_index"]))

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
            
            optimizer.zero_grad()
            loss_actions.backward()
            optimizer.step()

            loss_train_actions += loss_actions.item()
            acc, action_results_list = compute_action_acc(pred_actions, gt_actions)
            acc_train += acc
            for i, a in enumerate(action_results_list):
                action_total[i] += a

            iter_num += 1
            # validation
            model.eval()
            if iter_num % len(train_dataloader) == 0 and (not data_config["debug"]):    
                # evaluate the accuracy and loss on val set
                with torch.no_grad():
                    val_action_total = [0, 0, 0, 0, 0, 0, 0, 0]
                    for batch in tqdm(val_dataloader):
                        gt_actions = CUDA(batch["next_actions"])
                        pred_actions = model(CUDA(batch["img"]), CUDA(batch["bboxes"]), CUDA(batch["directions"]), CUDA(batch["priorities"]), CUDA(batch["edge_index"]))
                        
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
                
                print({
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

        print({
            'epoch': epoch,
            'learning rate': optimizer.state_dict()['param_groups'][0]['lr'],
            'loss_train_concept': loss_train_concepts,
            'loss_train_action': loss_train_actions,
            'acc_train': acc_train,
            'acc_train_weighted': action_weighted_acc,
        })

        if (epoch + 1) % 2 == 0 and (not data_config["debug"]):    
            if not os.path.exists("vis_input_weights/{}/{}".format(config['Data']['mode'], args.exp)):
                os.makedirs("vis_input_weights/{}/{}".format(config['Data']['mode'], args.exp))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_val,
            }, "vis_input_weights/{}/{}/{}_epoch{}_valacc{:.4f}.pth".format(config['Data']['mode'], args.exp, args.exp, epoch, val_action_weighted_acc))
            if best_acc < val_action_weighted_acc:
                best_acc = val_action_weighted_acc
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_val,
            }, "vis_input_weights/{}/{}/{}_best.pth".format(config['Data']['mode'], args.exp, args.exp))

    # wandb.finish()

if __name__ == "__main__":
    args = get_parser()
    # os.environ['WANDB__SERVICE_WAIT'] = "300"
    # os.environ['WANDB_API_KEY'] = 'f510977768bfee8889d74a65884aeec5f45a578f'
    if args.modular:
        train_modular(args)
    else:
        train_e2e(args)