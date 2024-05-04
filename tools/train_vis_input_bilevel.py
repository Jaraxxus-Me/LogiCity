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

def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default='config/tasks/Vis/ResNetNLM/easy_100_fixed_demo_bilevel.yaml', help='Path to the config file.')
    parser.add_argument("--exp", type=str, default='resnet_gnn_e2e')
    parser.add_argument("--implicit", action='store_true', help='Train the model in an Implicit Differentiation style.')
    parser.add_argument('--only_supervise_car', default=True, help='Only supervise the car actions.')
    return parser.parse_args()

def train_bilevel_unrolled(args):
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
        project = "logicity_vis_input_bilevel",
        name = "{}_{}".format(args.exp, config['Data']['mode']),
        config = config,
    )

    loss_ce = nn.CrossEntropyLoss()
    loss_bce = nn.BCELoss()

    wandb.watch(model)
    best_acc = -1
    epochs = config['Optimizer']['epochs'] * 2
    train_target = ['inner', 'outer'] * (epochs // 2)
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
            
            # do not cut the gradient for nlm here
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

            loss_concepts = loss_bce(pred_unary_concepts.reshape(-1), gt_unary_concepts.reshape(-1)) \
                            + loss_bce(pred_binary_concepts.reshape(-1), gt_binary_concepts.reshape(-1))
        
            if train_target[epoch] == 'inner':
                grounding_optimizer.zero_grad()
                loss_concepts.backward()
                grounding_optimizer.step()
            elif train_target[epoch] == 'outer':
                reasoning_optimizer.zero_grad()
                loss_actions.backward()
                reasoning_optimizer.step()

                # only accumulate loss and acc in outer loop
                loss_train_concepts += loss_concepts.item()
                loss_train_actions += loss_actions.item()
                acc, action_results_list = compute_action_acc(pred_actions, gt_actions)
                acc_train += acc
                for i, a in enumerate(action_results_list):
                    action_total[i] += a

                iter_num += 1
                # validation
                if iter_num % (len(train_dataloader)) == 0 and (not data_config["debug"]):    
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
                        (epoch-1)//2, iter_num, loss_val, acc_val))
                    
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
                        'iter': (epoch-1)//2*len(train_dataloader) + iter_num,
                        'loss_val': loss_val,
                        'acc_val': acc_val,
                        'acc_val_weighted': val_action_weighted_acc,
                    })

        if train_target[epoch] == 'outer':
            loss_train_concepts /= len(train_dataset)
            loss_train_actions /= len(train_dataset)
            acc_train /= len(train_dataset)
            print("Epoch: {}, Training Loss (Concepts): {:.4f}, Training Loss (Actions): {:.4f}, Sample Avg Acc: {:.4f}".format(
                (epoch-1)//2, loss_train_concepts, loss_train_actions, acc_train))
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
                'epoch': (epoch-1)//2,
                'learning rate (grounding)': grounding_optimizer.state_dict()['param_groups'][0]['lr'],
                'learning rate (reasoning)': reasoning_optimizer.state_dict()['param_groups'][0]['lr'],
                'loss_train_concept': loss_train_concepts,
                'loss_train_action': loss_train_actions,
                'acc_train': acc_train,
                'acc_train_weighted': action_weighted_acc,
            })

            if ((epoch-1)//2 + 1) % 2 == 0 and (not data_config["debug"]):    
                if not os.path.exists("vis_input_weights/{}/{}".format(config['Data']['mode'], args.exp)):
                    os.makedirs("vis_input_weights/{}/{}".format(config['Data']['mode'], args.exp))
                torch.save({
                    'epoch': (epoch-1)//2,
                    'model_state_dict': model.state_dict(),
                    'grounder_optimizer_state_dict': grounding_optimizer.state_dict(),
                    'reasoning_optimizer_state_dict': reasoning_optimizer.state_dict(),
                    'loss': loss_val,
                }, "vis_input_weights/{}/{}/{}_epoch{}_valacc{:.4f}.pth".format(config['Data']['mode'], args.exp, args.exp, (epoch-1)//2, val_action_weighted_acc))
                if best_acc < val_action_weighted_acc:
                    best_acc = val_action_weighted_acc
                    torch.save({
                    'epoch': (epoch-1)//2,
                    'model_state_dict': model.state_dict(),
                    'grounder_optimizer_state_dict': grounding_optimizer.state_dict(),
                    'reasoning_optimizer_state_dict': reasoning_optimizer.state_dict(),
                    'loss': loss_val,
                }, "vis_input_weights/{}/{}/{}_best.pth".format(config['Data']['mode'], args.exp, args.exp))

    wandb.finish()


def train_bilevel_implicit(args):
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
        project = "logicity_vis_input_bilevel",
        name = "{}_{}".format(args.exp, config['Data']['mode']),
        config = config,
    )

    loss_ce = nn.CrossEntropyLoss()
    loss_bce = nn.BCELoss()

    wandb.watch(model)
    best_acc = -1
    epochs = config['Optimizer']['epochs'] * 2
    train_target = ['inner', 'outer'] * (epochs // 2)
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
            
            # do not cut the gradient for nlm here
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

            loss_concepts = loss_bce(pred_unary_concepts.reshape(-1), gt_unary_concepts.reshape(-1)) \
                            + loss_bce(pred_binary_concepts.reshape(-1), gt_binary_concepts.reshape(-1))
        
            if train_target[epoch] == 'inner':
                grounding_optimizer.zero_grad()
                loss_concepts.backward()
                grounding_optimizer.step()
            elif train_target[epoch] == 'outer':
                reasoning_optimizer.zero_grad()
                # loss_actions.backward()
                # Here, grounding_params is: list(model.node_concept_predictor.parameters()) + list(model.edge_predictor.parameters())
                q = [torch.zeros_like(g) for g in grounding_params] # init_with_some_random_vector
                v = torch.autograd.grad(loss_actions, grounding_params, retain_graph=True, create_graph=True)
                lr_tmp = 1
                for _ in range(3):
                    Hq = torch.autograd.grad([torch.dot(v_i.flatten(), q_i.flatten()) for v_i, q_i in zip(v, q)], grounding_params, retain_graph=True, create_graph=True)
                    print(_, [torch.sum(Hq_i - v_i) for Hq_i, v_i in zip(Hq, v)])
                    q = [q_i - lr_tmp * (Hq_i - v_i) for q_i, Hq_i, v_i in zip(q, Hq, v)]

                v_ = torch.autograd.grad(loss_concepts, grounding_params, retain_graph=True, create_graph=True)
                Hq_ = torch.autograd.grad([torch.dot(v_i.flatten(), q_i.flatten()) for v_i, q_i in zip(v_, q)], \
                    model.reasoning_engine.parameters(), retain_graph=True, create_graph=True, allow_unused=True)
                d_U = torch.autograd.grad(loss_actions, model.reasoning_engine.parameters(), retain_graph=True, create_graph=True, allow_unused=True)
                
                # reasoning_optimizer.step()
                for i, param in enumerate(model.reasoning_engine.parameters()):
                    with torch.no_grad():
                        if d_U[i] != None and Hq_[i] != None:
                            param[:] = param - reasoning_opt_config["lr"] * (d_U[i] - Hq_[i])


                # only accumulate loss and acc in outer loop
                loss_train_concepts += loss_concepts.item()
                loss_train_actions += loss_actions.item()
                acc, action_results_list = compute_action_acc(pred_actions, gt_actions)
                acc_train += acc
                for i, a in enumerate(action_results_list):
                    action_total[i] += a

                iter_num += 1
                # validation
                if iter_num % (len(train_dataloader)) == 0 and (not data_config["debug"]):    
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
                        (epoch-1)//2, iter_num, loss_val, acc_val))
                    
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
                        'iter': (epoch-1)//2*len(train_dataloader) + iter_num,
                        'loss_val': loss_val,
                        'acc_val': acc_val,
                        'acc_val_weighted': val_action_weighted_acc,
                    })

        if train_target[epoch] == 'outer':
            loss_train_concepts /= len(train_dataset)
            loss_train_actions /= len(train_dataset)
            acc_train /= len(train_dataset)
            print("Epoch: {}, Training Loss (Concepts): {:.4f}, Training Loss (Actions): {:.4f}, Sample Avg Acc: {:.4f}".format(
                (epoch-1)//2, loss_train_concepts, loss_train_actions, acc_train))
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
                'epoch': (epoch-1)//2,
                'learning rate (grounding)': grounding_optimizer.state_dict()['param_groups'][0]['lr'],
                'learning rate (reasoning)': reasoning_optimizer.state_dict()['param_groups'][0]['lr'],
                'loss_train_concept': loss_train_concepts,
                'loss_train_action': loss_train_actions,
                'acc_train': acc_train,
                'acc_train_weighted': action_weighted_acc,
            })

            if ((epoch-1)//2 + 1) % 1 == 0 and (not data_config["debug"]):    
                if not os.path.exists("vis_input_weights/{}/{}".format(config['Data']['mode'], args.exp)):
                    os.makedirs("vis_input_weights/{}/{}".format(config['Data']['mode'], args.exp))
                torch.save({
                    'epoch': (epoch-1)//2,
                    'model_state_dict': model.state_dict(),
                    'grounder_optimizer_state_dict': grounding_optimizer.state_dict(),
                    'reasoning_optimizer_state_dict': reasoning_optimizer.state_dict(),
                    'loss': loss_val,
                }, "vis_input_weights/{}/{}/{}_epoch{}_valacc{:.4f}.pth".format(config['Data']['mode'], args.exp, args.exp, (epoch-1)//2, val_action_weighted_acc))
                if best_acc < val_action_weighted_acc:
                    best_acc = val_action_weighted_acc
                    torch.save({
                    'epoch': (epoch-1)//2,
                    'model_state_dict': model.state_dict(),
                    'grounder_optimizer_state_dict': grounding_optimizer.state_dict(),
                    'reasoning_optimizer_state_dict': reasoning_optimizer.state_dict(),
                    'loss': loss_val,
                }, "vis_input_weights/{}/{}/{}_best.pth".format(config['Data']['mode'], args.exp, args.exp))

    wandb.finish()


if __name__ == "__main__":
    args = get_parser()
    os.environ['WANDB__SERVICE_WAIT'] = "300"
    os.environ['WANDB_API_KEY'] = 'f510977768bfee8889d74a65884aeec5f45a578f'
    if args.implicit:
        train_bilevel_implicit(args)
    else:
        train_bilevel_unrolled(args)