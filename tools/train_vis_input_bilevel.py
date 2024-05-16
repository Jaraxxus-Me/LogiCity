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

def set_seed(seed):
    # seed init.
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)


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

    wandb.watch(model)
    best_acc = -1
    epochs = config['Optimizer']['epochs']
    for epoch in range(epochs):
        loss_train_actions, loss_val = 0., 0.
        acc_train, acc_val = 0., 0.

        action_total = [0, 0, 0, 0, 0, 0, 0, 0]

        iter_num = 0
        for batch in tqdm(train_dataloader):
            gt_actions = CUDA(batch["next_actions"])
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
        
            # inner optim
            grounding_optimizer.zero_grad()
            loss_actions.backward()
            grounding_optimizer.step()

            gt_actions = CUDA(batch["next_actions"])
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

            # outer optim
            reasoning_optimizer.zero_grad()
            loss_actions.backward()
            reasoning_optimizer.step()

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

        loss_train_actions /= len(train_dataset)
        acc_train /= len(train_dataset)
        print("Epoch: {}, Training Loss (Actions): {:.4f}, Sample Avg Acc: {:.4f}".format(
            epoch, loss_train_actions, acc_train))
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
            'loss_train_action': loss_train_actions,
            'acc_train': acc_train,
            'acc_train_weighted': action_weighted_acc,
        })

        if (epoch + 1) % 1 == 0 and (not data_config["debug"]):    
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


def train_bilevel_implicit(args):
    torch.set_printoptions(precision=16)
    torch.autograd.set_detect_anomaly(True)
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

    wandb.watch(model)
    best_acc = -1
    epochs = config['Optimizer']['epochs']
    for epoch in range(epochs):
        loss_train_actions, loss_val = 0., 0.
        acc_train, acc_val = 0., 0.

        action_total = [0, 0, 0, 0, 0, 0, 0, 0]

        iter_num = 0
        for batch in tqdm(train_dataloader):
            gt_actions = CUDA(batch["next_actions"])
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
        
            # inner optim
            grounding_optimizer.zero_grad()
            loss_actions.backward()
            grounding_optimizer.step()

            gt_actions = CUDA(batch["next_actions"])
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
            
            # outer optim
            reasoning_optimizer.zero_grad()
            d_U = torch.autograd.grad(loss_actions, model.reasoning_engine.parameters(), retain_graph=True, create_graph=True, allow_unused=True)
            q = []
            # grounding_params_selected = grounding_params
            # grounding_params_selected = [grounding_params[-2]] #singular
            grounding_params_selected = [grounding_params[-1]] # 4x4 
            # grounding_params_selected = [grounding_params[8]]
            for g in grounding_params_selected:
                q_i = torch.zeros_like(g, requires_grad=True).flatten()
                # q_i = torch.randn_like(g, requires_grad=True).flatten()
                # nn.init.kaiming_normal_(q_i.unsqueeze(0), mode="fan_out")
                q.append(q_i)
            q = torch.cat(q)
            v = torch.autograd.grad(loss_actions, grounding_params_selected, retain_graph=True, create_graph=True)
            v = torch.cat([v_i.flatten() for v_i in v])
            # v = torch.cat([g.grad.flatten() for g in grounding_params_selected])

            # when Hq = v, q = H^-1 * v
            #   0.5 * q.T * H * q - q.T * v
            # = 0.5 * q.T * v - q.T * v
            # = -0.5 * q.T * v
            # = -0.5 * (H^-1 * v).T * v
            # H_flat = CUDA(torch.tensor([]))
            # for v_i in v:
            #     H_flat = torch.cat((H_flat, torch.autograd.grad(v_i, grounding_params_selected, retain_graph=True)[0].flatten()))
            # with torch.no_grad():
            #     H = H_flat.view(grounding_params_selected[0].flatten().size()[0], -1)
            #     H_inverse = torch.inverse(H)
            #     print("Hessian:", H)
            #     print("Hessian -1:", H_inverse)
            #     print("Theoretical min:", -torch.dot(torch.mv(H_inverse, v), v)/2)
            #     print(torch.linalg.eig(H))
                # q = torch.mv(H_inverse, v)
                # Hq = torch.mv(H, q)
                # print("q:", q)
                # print("Hq:", Hq)
                # print("v:", v)

            # gradient descent
            lr_tmp = 10000
            for _ in range(5):
                Hq = torch.autograd.grad(torch.dot(v, q), grounding_params_selected, retain_graph=True, create_graph=True)
                Hq = torch.cat([Hq_i.flatten() for Hq_i in Hq])
                # print(Hq)
                with torch.no_grad():
                    # qHq = torch.dot(q, Hq)
                    # loss_tmp = qHq / 2 - torch.dot(q, v)
                    # loss_tmp = qHq - torch.dot(q, v) / 2
                    # print("epoch:", _, loss_tmp)
                    q = q - lr_tmp * (Hq - v)
                pass

            # conjugate gradient
            # Hq = torch.autograd.grad(torch.dot(v, q), grounding_params_selected, retain_graph=True)
            # Hq = torch.cat([Hq_i.flatten() for Hq_i in Hq])
            # d = v - Hq
            # r = d.clone().detach()
            # for _ in range(801805):
            #     Hd = torch.autograd.grad(torch.dot(v, d), grounding_params_selected, retain_graph=True)
            #     Hd = torch.cat([Hd_i.flatten() for Hd_i in Hd])
            #     Hq = torch.autograd.grad(torch.dot(v, q), grounding_params_selected, retain_graph=True)
            #     Hq = torch.cat([Hq_i.flatten() for Hq_i in Hq])
            #     with torch.no_grad():
            #         dHd = torch.dot(d, Hd)
            #         qHq = torch.dot(q, Hq)
            #         rr = torch.dot(r, r)
            #         # if _%1024==0:
            #         if True:
            #             loss_tmp = qHq / 2 - torch.dot(q, v)
            #             print("epoch:", _, loss_tmp)
            #             print(rr)
            #         a = rr / dHd
            #         q = q + a * d
            #         r = r - a * Hd
            #         b = torch.dot(r, r) / rr
            #         d = r + b * d

            # v_ = torch.autograd.grad(loss_actions, grounding_params, retain_graph=True, create_graph=True)
            v_ = v
            Hq_ = torch.autograd.grad(torch.dot(v_, q), model.reasoning_engine.parameters(), retain_graph=True, create_graph=True, allow_unused=True)
            
            # reasoning_optimizer.step()
            for i, param in enumerate(model.reasoning_engine.parameters()):
                with torch.no_grad():
                    if d_U[i] != None and Hq_[i] != None:
                        param[:] = param - reasoning_opt_config["lr"] * (d_U[i] - Hq_[i])
                    else:
                        # print(i)
                        pass

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

        loss_train_actions /= len(train_dataset)
        acc_train /= len(train_dataset)
        print("Epoch: {}, Training Loss (Actions): {:.4f}, Sample Avg Acc: {:.4f}".format(
            epoch, loss_train_actions, acc_train))
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
            'loss_train_action': loss_train_actions,
            'acc_train': acc_train,
            'acc_train_weighted': action_weighted_acc,
        })

        if (epoch + 1) % 1 == 0 and (not data_config["debug"]):    
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


if __name__ == "__main__":
    args = get_parser()
    os.environ['WANDB__SERVICE_WAIT'] = "300"
    os.environ['WANDB_API_KEY'] = 'f510977768bfee8889d74a65884aeec5f45a578f'

    import time
    seed = int(time.time())
    # seed = 176
    print("seed:", seed)
    set_seed(seed)

    if args.implicit:
        train_bilevel_implicit(args)
    else:
        train_bilevel_unrolled(args)