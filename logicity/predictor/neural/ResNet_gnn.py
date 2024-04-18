import torch
import yaml
import torch.nn as nn
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch_geometric.nn import NNConv


class LogicityPredictorVis(nn.Module):
    def __init__(self, mode):
        super(LogicityPredictorVis, self).__init__()

        # Build feature extractor
        self.resnet_layer_num = 4
        self.resnet_fpn = resnet_fpn_backbone(
            "resnet50", pretrained=True, trainable_layers=self.resnet_layer_num+1
        )
        self.img_feature_channels = self.resnet_fpn.out_channels * self.resnet_layer_num # 1024
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Process ontology
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

        # Build node concept predictor
        self.node_channels = len(self.node_concept_names)*256
        self.node_concept_predictor = nn.Sequential(
            nn.Linear(self.img_feature_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.node_channels),
            nn.Sigmoid(),
        )

        # Build edge concept predictor
        # node attributes used for edge prediction: bbox + direction
        self.bbox_channels = 4 + 4
        self.BBOX_POS_MAX = 1024
        # HigherPri can be directly calc by priority in nodes, no need to predict
        assert "HigherPri" in self.edge_concept_names
        self.edge_channels = len(self.edge_concept_names) - 1
        self.action_channels = len(self.action_names)
        if mode == "easy":
            self.max_node_num = 8 # TODO: how to get this num from config file? or just hard code it?
        else:
            pass
        self.edge_predictor = nn.Sequential(
            nn.Linear(self.bbox_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, (self.max_node_num-1)*self.edge_channels),
            nn.Sigmoid(),
        )

        # Build action predictor
        self.edge_processor = nn.Sequential(
            nn.Linear(self.edge_channels+1, 128),
            nn.ReLU(),
            nn.Linear(128, self.node_channels*self.action_channels)
        ) # A neural network that maps edge features edge_attr of shape [-1, num_edge_features] to shape [-1, in_channels * out_channels]
        self.gnn = NNConv(in_channels=self.node_channels, out_channels=self.action_channels, nn=self.edge_processor) # don't support batch operation


    def forward(self, batch_imgs, batch_bboxes, batch_directions, batch_priorities):
        device = batch_imgs.device
        B = batch_imgs.shape[0]
        N = batch_bboxes.shape[1]
        batch_imgs = batch_imgs.permute(0, 3, 2, 1) # Bx3xHxW
        # normalize with mean and std of ImageNet
        batch_imgs = self.transform(batch_imgs)
        
        # Extract img features
        with torch.no_grad(): # frozen
            fpn_features = self.resnet_fpn(batch_imgs)
        feature_list = []
        for layer in range(self.resnet_layer_num):
            feature = fpn_features[str(layer)]
            feature = torch.nn.functional.interpolate(feature, fpn_features["0"].shape[-2:], mode="bilinear")
            feature_list.append(feature)
        imgs_features = torch.cat(feature_list, dim=1)
        imgs_features = torch.nn.functional.interpolate(imgs_features, batch_imgs.shape[-2:], mode="bilinear") # B x C_img x H x W
        
        # Get bbox features
        bboxes = [batch_bboxes[i] for i in range(batch_bboxes.shape[0])]
        roi_features = torchvision.ops.roi_pool(imgs_features, bboxes, output_size=1).squeeze()
        roi_features = roi_features.view(B, N, roi_features.shape[1]) # B x N x C_img
        
        # Predict node concepts
        node_concepts = self.node_concept_predictor(roi_features) # B x N x (concept_num x 256)
        
        # Create scene graph (node:(concept, bbox, direction, priority))
        # 1. concat node attributes use for edge prediction (bbox, direction)
        node_attributes = torch.cat([batch_bboxes/self.BBOX_POS_MAX, batch_directions], dim=-1) # B x N x (4+4)
        # 2. prepare edge idxs
        edge_idxs = []
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                else:
                    edge_idxs.append([i, j])
        edge_idxs = torch.Tensor(edge_idxs).permute(1, 0).to(torch.int64).to(device)
        # 3. predict edge attributes
        edge_attributes = self.edge_predictor(node_attributes) # B x N x ((N-1) x C_edge)
        edge_attributes = edge_attributes.view(B, -1, self.edge_channels) # B x (N x (N-1)) x C_edge
        # 4. add HigherPri to edge attributes
        pri_mask = (batch_priorities.unsqueeze(2)>batch_priorities.unsqueeze(1)).to(torch.float32)
        higher_pri = torch.zeros(B, N, N-1)
        higher_pri[:, :, :N-1] = pri_mask[:, :, :N-1] 
        higher_pri[:, :, N-1:] = pri_mask[:, :, N:]
        higher_pri = higher_pri.view(B, -1).to(device)
        edge_attributes = torch.cat([edge_attributes, higher_pri.unsqueeze(-1)], dim=-1) # B x (N x (N-1)) x (C_edge+1)

        # Predict actions
        next_actions = self.gnn(node_concepts[0], edge_idxs, edge_attributes[0])

        return next_actions