import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import wandb
import json
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
    parser.add_argument("--model", type=str, default='LogicityPredictorVis', help="model name")
    parser.add_argument("--data_path", type=str, default='vis_dataset/easy_200')
    parser.add_argument("--mode", type=str, default='easy')
    parser.add_argument("--exp", type=str, default='easy_200')
    return parser.parse_args()

    
def compute_action_acc(pred, label):
    pred = CPU(pred)
    label = CPU(label)
    pred = np.argmax(pred, axis=-1)
    label = np.argmax(label, axis=-1)
    acc = np.sum(pred == label) / len(label)
    # print(pred, label, acc)
    return acc

if __name__ == "__main__":
    args = get_parser()
    vis_dataset_path = args.data_path
    mode = args.mode
    exp_name = args.exp
    lr = 5e-6
    epochs = 200

    config = {
        "dataset": "easy_200",
        "epochs": epochs,
        "learning_rate": lr,
    }

    wandb.init(
        project = "logicity_vis_input",
        name = "resnet_gnn_{}".format(json.dumps(config)),
        config = config,
    )

    model = MODEL_BUILDER[args.model](mode)
    model = CUDA(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # prepare train, val, and test set
    train_vis_dataset_path = os.path.join(vis_dataset_path, "train/train_{}.pkl".format(exp_name))
    val_vis_dataset_path = os.path.join(vis_dataset_path, "val/val_{}.pkl".format(exp_name))
    test_vis_dataset_path = os.path.join(vis_dataset_path, "test/test_{}.pkl".format(exp_name))
    train_dataset = VisDataset(train_vis_dataset_path, batch_size=1)
    train_dataloader = DataLoader(train_dataset, batch_size=1)
    val_dataset = VisDataset(val_vis_dataset_path, batch_size=1)
    val_dataloader = DataLoader(val_dataset, batch_size=1)
    test_dataset = VisDataset(test_vis_dataset_path, batch_size=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    loss_ce = nn.CrossEntropyLoss()

    wandb.watch(model)
    best_acc = -1
    for epoch in range(epochs):
        loss_train, loss_val = 0., 0.
        acc_train, acc_val = 0., 0.

        for batch in tqdm(train_dataloader):
            gt_actions = CUDA(batch["next_actions"][0])
            pred_actions = model(CUDA(batch["img"]), CUDA(batch["bboxes"]), CUDA(batch["directions"]), CUDA(batch["priorities"]))
            loss_actions = loss_ce(pred_actions, gt_actions)

            loss = loss_actions
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()
            loss_train += loss.item()
            acc = compute_action_acc(pred_actions, gt_actions)
            acc_train += acc

        loss_train /= len(train_dataset)
        acc_train /= len(train_dataset)
        print("Epoch: {},Training Loss: {:.4f}, Acc: {:.4f}".format(
            epoch, loss_train, acc_train))

        # evaluate the accuracy and loss on val set
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                gt_actions = CUDA(batch["next_actions"][0])
                pred_actions = model(CUDA(batch["img"]), CUDA(batch["bboxes"]), CUDA(batch["directions"]), CUDA(batch["priorities"]))
                loss_actions = loss_ce(pred_actions, gt_actions)
                loss = loss_actions
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
            }, "vis_input_weights/resnet_gnn_lr{}_epoch{}_valacc{:.3f}.pth".format(lr, epoch, acc_val))
            if best_acc < acc_val:
                best_acc = acc_val
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_val,
                }, "vis_input_weights/resnet_gnn_best.pth")

    # evaluate the accuracy and loss on test set
    loss_test = 0.
    acc_test = 0.
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            gt_actions = CUDA(batch["next_actions"][0])
            pred_actions = model(CUDA(batch["img"]), CUDA(batch["bboxes"]), CUDA(batch["directions"]), CUDA(batch["priorities"]))
            loss_actions = loss_ce(pred_actions, gt_actions)
            loss = loss_actions
            loss_test += loss.item()
            acc = compute_action_acc(pred_actions, gt_actions)
            acc_test += acc
    
    loss_test /= len(test_dataset)
    acc_test /= len(test_dataset)
    print("Epoch: {}, Testing Loss: {:.4f}, Acc: {:.4f}".format(
        epoch, loss_test, acc_test))





    wandb.finish()