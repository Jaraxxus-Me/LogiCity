import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import argparse
import yaml
from logicity.utils.dataset import VisDataset

def CPU(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

def CUDA(x):
    return x.cuda()

def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=str, default='vis_dataset/easy_1k/easy_1k_5.pkl')
    parser.add_argument("--mode", type=str, default='easy')
    return parser.parse_args()

# TODO
# This bilevel optimization model uses an img as vis input and output agents' next actions
class LogicityPredictorVis(nn.Module):
    def __init__(self, mode):
        super(LogicityPredictorVis, self).__init__()
        # build feature extractor
        self.resnet_layer_num = 4
        self.resnet_fpn = resnet_fpn_backbone(
            "resnet50", pretrained=True, trainable_layers=self.resnet_layer_num+1
        )
        self.img_feature_channels = self.resnet_fpn.out_channels * self.resnet_layer_num # 1024
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # process ontology
        assert mode in ["easy", "medium", "hard", "expert"]
        if mode in ["easy", "medium"]:
            ontology_yaml_file = "config/rules/ontology_{}.yaml".format(mode)
        else:
            ontology_yaml_file = "config/rules/ontology_full.yaml"
        with open(ontology_yaml_file, 'r') as file:
            self.ontology_config = yaml.load(file, Loader=yaml.Loader)
        self.node_concept_names = []
        self.edge_concept_names = []
        self.action_names = []
        for predicate in self.ontology_config["Predicates"]:
            predicate_name = list(predicate.keys())[0]
            if predicate_name.startswith("Is"):
                self.node_concept_names.append(predicate_name)
            elif predicate[predicate_name]["arity"] == 2:
                self.edge_concept_names.append(predicate_name)
            else:
                self.action_names.append(predicate_name)
        # build node concept predictor
        self.multilabel_classifier = nn.Sequential(
            nn.Linear(self.img_feature_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, len(self.node_concept_names)),
            nn.Sigmoid(),
        )
        
    def forward(self, batch_imgs, batch_bboxes):
        B = batch_imgs.shape[0]
        N = batch_bboxes.shape[1]
        batch_imgs = batch_imgs.permute(0, 3, 2, 1) # Bx3xHxW
        # normalize with mean and std of ImageNet
        batch_imgs = self.transform(batch_imgs)
        # extract img features
        with torch.no_grad(): # frozen
            fpn_features = self.resnet_fpn(batch_imgs)
        feature_list = []
        for layer in range(self.resnet_layer_num):
            feature = fpn_features[str(layer)]
            feature = torch.nn.functional.interpolate(feature, fpn_features["0"].shape[-2:], mode="bilinear")
            feature_list.append(feature)
        imgs_features = torch.cat(feature_list, dim=1)
        imgs_features = torch.nn.functional.interpolate(imgs_features, batch_imgs.shape[-2:], mode="bilinear") # BxCxHxW
        # get bbox features
        bboxes = [batch_bboxes[i] for i in range(batch_bboxes.shape[0])]
        roi_features = torchvision.ops.roi_pool(imgs_features, bboxes, output_size=1).squeeze()
        roi_features = roi_features.view(B, N, roi_features.shape[1]) # BxNxC
        # predict concepts
        bbox_concepts = self.multilabel_classifier(roi_features)
        # create scene graph (node:(pos, concept, priority))
        # ...
        # predict actions
        # ...
        return 0
    
def compute_action_acc(pred, label):
    pred = CPU(pred).numpy()
    label = CPU(label).numpy()
    pred = np.argmax(pred, axis=-1)
    acc = np.sum(pred == label) / len(label)
    # print(pred, label, acc)
    return acc

if __name__ == "__main__":
    args = get_parser()
    vis_dataset_path = args.data_path
    mode = args.mode
    model = LogicityPredictorVis(mode)
    model = CUDA(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # split into train and test set
    dataset = VisDataset(vis_dataset_path)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    loss_ce = nn.CrossEntropyLoss()
    for epoch in range(1000):
        loss_train, loss_test = 0., 0.
        acc_train, acc_test = 0., 0.

        for batch in train_dataset:
            gt_actions = CUDA(batch["next_actions"].flatten(0))
            pred_actions = model(CUDA(batch["imgs"]), CUDA(batch["bboxes"]))
            loss_actions = loss_ce(pred_actions, gt_actions)
            loss = loss_actions
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_train += loss.item()
            acc = compute_action_acc(pred_actions, gt_actions)
            acc_train += acc
        
        loss_train /= len(train_dataset)
        acc_train /= len(train_dataset)
        print("Training: {}, Loss: {:.4f}, Acc: {:.4f}".format(
            epoch, loss_train, acc_train))
        
        # evaluate the accuracy and loss on test set
        with torch.no_grad():
            for batch in test_dataset:
                gt_actions = np.array([list(actions.values()) for actions in batch["next_actions"]]).flatten()
                gt_actions = torch.from_numpy(gt_actions)
                pred_actions = model(batch["imgs"], batch["bboxes"]) # cuda?
                loss_actions = loss_ce(pred_actions, gt_actions)
                loss = loss_actions
                loss_test += loss.item()
                acc = compute_action_acc(pred_actions, gt_actions)
                acc_test += acc
        
        loss_test /= len(test_dataset)
        acc_test /= len(test_dataset)
        print("Testing: {}, Loss: {:.4f}, Acc: {:.4f}".format(
            epoch, loss_test, acc_test))
