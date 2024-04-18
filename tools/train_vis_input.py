import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import wandb
import json
from logicity.utils.dataset import VisDataset
from logicity.predictor import MODEL_BUILDER

def CPU(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

def CUDA(x):
    return x.cuda()

def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default='LogicityPredictorVis', help="model name")
    parser.add_argument("--data_path", type=str, default='vis_dataset/easy_1k/easy_1k_5.pkl')
    parser.add_argument("--mode", type=str, default='easy')
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
    lr = 5e-6
    epochs = 100

    config = {
        "dataset": "easy_1k_5",
        "epochs": epochs,
        "learning_rate": lr,
    }

    wandb.init(
        project = "logicity_vis_input",
        name = "baseline(mlp+gnn)_{}".format(json.dumps(config)),
        config = config,
    )

    model = MODEL_BUILDER[args.model](mode)
    model = CUDA(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # split into train and test set
    dataset = VisDataset(vis_dataset_path, batch_size=1)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    loss_ce = nn.CrossEntropyLoss()

    wandb.watch(model)
    for epoch in range(epochs):
        loss_train, loss_test = 0., 0.
        acc_train, acc_test = 0., 0.

        for batch in tqdm(train_dataset):
            gt_actions = CUDA(batch["next_actions"][0])
            pred_actions = model(CUDA(batch["imgs"]), CUDA(batch["bboxes"]), CUDA(batch["directions"]), CUDA(batch["priorities"]))
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

        # evaluate the accuracy and loss on test set
        with torch.no_grad():
            for batch in tqdm(test_dataset):
                gt_actions = CUDA(batch["next_actions"][0])
                pred_actions = model(CUDA(batch["imgs"]), CUDA(batch["bboxes"]), CUDA(batch["directions"]), CUDA(batch["priorities"]))
                loss_actions = loss_ce(pred_actions, gt_actions)
                loss = loss_actions
                loss_test += loss.item()
                acc = compute_action_acc(pred_actions, gt_actions)
                acc_test += acc
        
        loss_test /= len(test_dataset)
        acc_test /= len(test_dataset)
        print("Epoch: {}, Testing Loss: {:.4f}, Acc: {:.4f}".format(
            epoch, loss_test, acc_test))
        
        wandb.log({
            'epoch': epoch,
            'learning rate': optimizer.state_dict()['param_groups'][0]['lr'],
            'loss_train': loss_train,
            'loss_test': loss_test,
            'acc_train': acc_train,
            'acc_test': acc_test,
        })

        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_train,
            }, "vis_input_weights/baseline_lr{}_epoch{}_valacc{:.3f}.pth".format(lr, epoch, acc_test))

    wandb.finish()