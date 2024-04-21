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
    parser.add_argument("--exp", type=str, default='resnet_gnn')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--add_concept_loss", action='store_true', help='Add concept_loss in addition to action_loss.')
    parser.add_argument('--bilevel', action='store_true', help='Train the model in a bilevel style.')
    return parser.parse_args()

def compute_action_acc(pred, label):
    pred = CPU(pred)
    label = CPU(label)
    pred = np.argmax(pred, axis=-1)
    label = np.argmax(label, axis=-1)
    acc = np.sum(pred == label) / len(label)
    # print(pred, label, acc)
    return acc

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
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # prepare train, val, and test set
    train_vis_dataset_path = os.path.join(vis_dataset_path, "train/train_{}.pkl".format(dataset_name))
    val_vis_dataset_path = os.path.join(vis_dataset_path, "val/val_{}.pkl".format(dataset_name))
    test_vis_dataset_path = os.path.join(vis_dataset_path, "test/test_{}.pkl".format(dataset_name))
    train_dataset = VisDataset(train_vis_dataset_path)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataset = VisDataset(val_vis_dataset_path)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_dataset = VisDataset(test_vis_dataset_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    loss_ce = nn.CrossEntropyLoss()
    loss_bce = nn.BCELoss()

    wandb.watch(model)
    best_acc = -1
    for epoch in range(epochs):
        loss_train, loss_val = 0., 0.
        acc_train, acc_val = 0., 0.

        for batch in tqdm(train_dataloader):
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
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()
            loss_train += loss.item()
            acc = compute_action_acc(pred_actions, gt_actions)
            acc_train += acc

        loss_train /= len(train_dataset)
        acc_train /= len(train_dataset)
        print("Epoch: {},Training Loss: {:.4f}, Acc: {:.4f}, lr: {}".format(
            epoch, loss_train, acc_train, optimizer.state_dict()['param_groups'][0]['lr'],))

        # evaluate the accuracy and loss on val set
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
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
                loss_val += loss.item()
                acc = compute_action_acc(pred_actions, gt_actions)
                acc_val += acc
        
        loss_val /= len(val_dataset)
        acc_val /= len(val_dataset)
        print("Epoch: {}, Validation Loss: {:.4f}, Acc: {:.4f}".format(
            epoch, loss_val, acc_val))

        wandb.log({
            'epoch': epoch,
            'learning rate': optimizer.state_dict()['param_groups'][0]['lr'],
            'loss_train': loss_train,
            'loss_val': loss_val,
            'acc_train': acc_train,
            'acc_val': acc_val,
        })

        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_val,
            }, "vis_input_weights/{}/{}_lr{}_epoch{}_valacc{:.4f}.pth".format(mode, args.exp, lr, epoch, acc_val))
            if best_acc < acc_val:
                best_acc = acc_val
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_val,
                }, "vis_input_weights/{}/{}_best.pth".format(mode, args.exp))

    wandb.finish()

def train_bilevel(args):
    pass

if __name__ == "__main__":
    args = get_parser()
    if args.bilevel:
        train_bilevel(args)
    else:
        train(args)