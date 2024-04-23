import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import wandb
import os
from logicity.utils.dataset import VisDataset
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


def CPU(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

def CUDA(x):
    return x.cuda()

def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default='LogicityVisPredictorGNN', help="model name")
    parser.add_argument("--data_path", type=str, default='vis_dataset/expert_200_fixed')
    parser.add_argument("--mode", type=str, default='expert')
    parser.add_argument("--dataset_name", type=str, default='expert_200_fixed')
    parser.add_argument("--exp", type=str, default='resnet_gnn')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--add_concept_loss", action='store_true', help='Add concept_loss in addition to action_loss.')
    parser.add_argument('--bilevel', action='store_true', help='Train the model in a bilevel style.')
    parser.add_argument('--only_supervise_car', action='store_true', help='Only supervise the actions of car.')
    return parser.parse_args()

def compute_action_acc(pred, label):
    pred = CPU(pred)
    label = CPU(label)
    pred = np.argmax(pred, axis=-1)
    label = np.argmax(label, axis=-1)
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

def train(args):
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # prepare train and val set
    train_vis_dataset_path = os.path.join(vis_dataset_path, "train/train_{}.pkl".format(dataset_name))
    val_vis_dataset_path = os.path.join(vis_dataset_path, "val/val_{}.pkl".format(dataset_name))
    train_dataset = VisDataset(train_vis_dataset_path)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataset = VisDataset(val_vis_dataset_path)
    val_dataloader = DataLoader(val_dataset, batch_size=1)

    loss_ce = nn.CrossEntropyLoss()
    loss_bce = nn.BCELoss()

    wandb.watch(model)
    best_acc = -1
    for epoch in range(epochs):
        loss_train, loss_val = 0., 0.
        acc_train, acc_val = 0., 0.

        action_total = [0, 0, 0, 0, 0, 0, 0, 0]

        iter_num = 0
        for batch in tqdm(train_dataloader):
            gt_actions = CUDA(batch["next_actions"][0])
            gt_unary_concepts = CUDA(batch["predicates"]["unary"][0])
            gt_binary_concepts = CUDA(batch["predicates"]["binary"][0])
            # TODO: for zhaoyu, gt_binary_concepts[:, -1] is the concept of "sees", use 11x10 matrix to represent the edge
            edge = gt_binary_concepts[:, -1].reshape(-1, 11, 10)
            pred_actions, pred_unary_concepts, pred_binary_concepts = model(CUDA(batch["img"]), CUDA(batch["bboxes"]), CUDA(batch["directions"]), CUDA(batch["priorities"]))
            
            if args.only_supervise_car:
                gt_types = batch["types"]
                gt_action_labels = np.argmax(CPU(gt_actions), axis=-1)
                is_car_mask = []
                for type in gt_types:
                    is_car_mask.append((type[0] == "Car"))
                car_gt_num = np.sum(is_car_mask)
                car_action_gt_num_list = []
                for i in range(4):
                    car_action_gt_num_list.append(np.sum((gt_action_labels == i) & is_car_mask))
                car_action_gt_nums = torch.FloatTensor(car_action_gt_num_list)
                class_weights = car_gt_num / (car_action_gt_nums + 1e-6)
                car_idxs = np.where(np.array(is_car_mask)==True)[0].tolist()
                loss_ce = nn.CrossEntropyLoss(weight=CUDA(class_weights))
                pred_actions = pred_actions[car_idxs]
                gt_actions = gt_actions[car_idxs]
            loss_actions = loss_ce(pred_actions, gt_actions)
            loss = loss_actions
            if args.add_concept_loss:
                loss_concepts = loss_bce(pred_unary_concepts, gt_unary_concepts) \
                                + loss_bce(pred_binary_concepts, gt_binary_concepts)
                loss += loss_concepts
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_train += loss.item()
            acc, action_results_list = compute_action_acc(pred_actions, gt_actions)
            acc_train += acc
            for i, a in enumerate(action_results_list):
                action_total[i] += a

            iter_num += 1
            # validation
            if iter_num % (len(train_dataloader)//5) == 0:
                # evaluate the accuracy and loss on val set
                with torch.no_grad():
                    val_action_total = [0, 0, 0, 0, 0, 0, 0, 0]
                    for batch in tqdm(val_dataloader):
                        gt_actions = CUDA(batch["next_actions"][0])
                        gt_unary_concepts = CUDA(batch["predicates"]["unary"][0])
                        gt_binary_concepts = CUDA(batch["predicates"]["binary"][0])
                        pred_actions, pred_unary_concepts, pred_binary_concepts = model(CUDA(batch["img"]), CUDA(batch["bboxes"]), CUDA(batch["directions"]), CUDA(batch["priorities"]))
                        
                        if args.only_supervise_car:
                            gt_types = batch["types"]
                            gt_action_labels = np.argmax(CPU(gt_actions), axis=-1)
                            is_car_mask = []
                            for type in gt_types:
                                is_car_mask.append((type[0] == "Car"))
                            car_gt_num = np.sum(is_car_mask)
                            car_action_gt_num_list = []
                            for i in range(4):
                                car_action_gt_num_list.append(np.sum((gt_action_labels == i) & is_car_mask))
                            car_action_gt_nums = torch.FloatTensor(car_action_gt_num_list)
                            class_weights = car_gt_num / (car_action_gt_nums + 1e-6)
                            car_idxs = np.where(np.array(is_car_mask)==True)[0].tolist()
                            loss_ce = nn.CrossEntropyLoss(weight=CUDA(class_weights))
                            pred_actions = pred_actions[car_idxs]
                            gt_actions = gt_actions[car_idxs]
                        loss_actions = loss_ce(pred_actions, gt_actions)
                        loss = loss_actions
                        if args.add_concept_loss:
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

        loss_train /= len(train_dataset)
        acc_train /= len(train_dataset)
        print("Epoch: {},Training Loss: {:.4f}, Sample Avg Acc: {:.4f}, lr: {}".format(
            epoch, loss_train, acc_train, optimizer.state_dict()['param_groups'][0]['lr']))
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
            'learning rate': optimizer.state_dict()['param_groups'][0]['lr'],
            'loss_train': loss_train,
            'acc_train': acc_train,
            'acc_train_weighted': action_weighted_acc,
        })

        # if (epoch + 1) % 5 == 0:
        if True:
            if not os.path.exists("vis_input_weights/{}/{}".format(mode, args.exp)):
                os.makedirs("vis_input_weights/{}/{}".format(mode, args.exp))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_val,
            }, "vis_input_weights/{}/{}/{}_lr{}_epoch{}_valacc{:.4f}.pth".format(mode, args.exp, args.exp, lr, epoch, acc_val))
            if best_acc < acc_val:
                best_acc = acc_val
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_val,
                }, "vis_input_weights/{}/{}/{}_best.pth".format(mode, args.exp, args.exp))

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
    train_dataset = VisDataset(train_vis_dataset_path)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataset = VisDataset(val_vis_dataset_path)
    val_dataloader = DataLoader(val_dataset, batch_size=1)

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
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                gt_actions = CUDA(batch["next_actions"][0])
                gt_unary_concepts = CUDA(batch["predicates"]["unary"][0])
                gt_binary_concepts = CUDA(batch["predicates"]["binary"][0])
                pred_actions, pred_unary_concepts, pred_binary_concepts = model(CUDA(batch["img"]), CUDA(batch["bboxes"]), CUDA(batch["directions"]), CUDA(batch["priorities"]))
                loss_actions = loss_ce(pred_actions, gt_actions)
                loss_concepts = loss_bce(pred_unary_concepts, gt_unary_concepts) \
                                + loss_bce(pred_binary_concepts, gt_binary_concepts)
                action_loss_val += loss_actions.item()
                concept_loss_val += loss_concepts.item()
                acc = compute_action_acc(pred_actions, gt_actions)
                acc_val += acc
        
        action_loss_val /= len(val_dataset)
        concept_loss_val /= len(val_dataset)
        acc_val /= len(val_dataset)
        print("Epoch: {}, Validation Action Loss: {:.4f}, Concept Loss: {:.4f}, Acc: {:.4f}".format(
            epoch, action_loss_val, concept_loss_val, acc_val))

        wandb.log({
            'epoch': epoch,
            'learning rate': cur_lr,
            'action_loss_train': action_loss_train,
            'concept_loss_train': concept_loss_train,
            'action_loss_val': action_loss_val,
            'concept_loss_val': concept_loss_val,
            'acc_train': acc_train,
            'acc_val': acc_val,
        })

        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'inner_optimizer_state_dict': inner_optimizer.state_dict(),
                'outer_optimizer_state_dict': outer_optimizer.state_dict(),
                'action_loss_val': action_loss_val,
                'concept_loss_val': concept_loss_val,
            }, "vis_input_weights/{}/{}_lr{}_epoch{}_valacc{:.4f}.pth".format(mode, args.exp, lr, epoch, acc_val))
            if best_acc < acc_val:
                best_acc = acc_val
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'inner_optimizer_state_dict': inner_optimizer.state_dict(),
                    'outer_optimizer_state_dict': outer_optimizer.state_dict(),
                    'action_loss_val': action_loss_val,
                    'concept_loss_val': concept_loss_val,
                }, "vis_input_weights/{}/{}_best.pth".format(mode, args.exp))

    wandb.finish()

if __name__ == "__main__":
    args = get_parser()
    if args.bilevel:
        train_bilevel(args)
    else:
        train(args)