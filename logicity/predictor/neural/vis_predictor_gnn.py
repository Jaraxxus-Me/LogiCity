import torch
import yaml
import torch.nn as nn
from torch_geometric.nn import NNConv
from logicity.predictor.neural.resnet_fpn import LogicityFeatureExtractor


class LogicityVisReasoningEngine(nn.Module):
    def __init__(self, mode, img_feature_channels):
        super(LogicityVisReasoningEngine, self).__init__()

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
            nn.Linear(img_feature_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.node_channels),
        )
        self.node_concept_interpreter = nn.Sequential(
            nn.Linear(self.node_channels, len(self.node_concept_names)),
            nn.Sigmoid(),
        )

        # Build edge concept predictor
        # node attributes used for edge prediction: bbox + direction
        self.bbox_channels = 4 + 4
        self.BBOX_POS_MAX = 1024
        # HigherPri can be directly calc by priority in nodes, no need to predict
        assert "HigherPri" in self.edge_concept_names
        self.edge_channels = len(self.edge_concept_names) - 1
        self.edge_predictor = nn.Sequential(
            nn.Linear(2*self.bbox_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.edge_channels),
            nn.Sigmoid(),
        )

        # Build action predictor
        self.action_channels = len(self.action_names)
        self.edge_processor = nn.Sequential(
            nn.Linear(self.edge_channels+1, 128),
            nn.ReLU(),
            nn.Linear(128, self.node_channels*self.action_channels)
        ) # A neural network that maps edge features edge_attr of shape [-1, num_edge_features] to shape [-1, in_channels * out_channels]
        self.gnn = NNConv(in_channels=self.node_channels, out_channels=self.action_channels, nn=self.edge_processor) # don't support batch operation

    def forward(self, roi_features, batch_bboxes, batch_directions, batch_priorities):
        device = roi_features.device
        B = roi_features.shape[0]
        N = roi_features.shape[1]
       
        # Predict node concepts
        node_concepts = self.node_concept_predictor(roi_features) # B x N x (concept_num x 256)
        node_concepts_explicit = self.node_concept_interpreter(node_concepts)
        
        # Create scene graph (node:(concept, bbox, direction, priority))
        # 1. concat node attributes use for edge prediction (bbox, direction)
        node_attributes = torch.cat([batch_bboxes/self.BBOX_POS_MAX, batch_directions], dim=-1) # B x N x (4+4)
        # 2. prepare edge idxs
        # TODO: optimize this part, maybe use the sees matrix for edge_idxs? also, batched operation?
        tmp_idx = torch.arange(N)
        i, j = torch.meshgrid(tmp_idx, tmp_idx, indexing='ij')
        tmp_mask = (i != j)
        edge_idxs = torch.stack((i[tmp_mask],j[tmp_mask]),dim=1).T.to(device) # 2 x (N x (N-1))
        # 3. predict edge attributes
        node_attributes_paired = torch.zeros(B, N, N-1, 16).to(device)
        upper_pairing_idxs = torch.triu_indices(N, N, offset=1).to(device)
        lower_pairing_idxs = torch.tril_indices(N, N, offset=-1).to(device)
        node_attributes_paired[:, upper_pairing_idxs[0], upper_pairing_idxs[1]-1] = torch.cat([
            node_attributes[:, upper_pairing_idxs[0]], node_attributes[:, upper_pairing_idxs[1]]
        ], dim=-1)
        node_attributes_paired[:, lower_pairing_idxs[0], lower_pairing_idxs[1]] = torch.cat([
            node_attributes[:, lower_pairing_idxs[0]], node_attributes[:, lower_pairing_idxs[1]]
        ], dim=-1)
        node_attributes_paired = node_attributes_paired.view(B, (N*(N-1)), -1)
        edge_attributes = self.edge_predictor(node_attributes_paired) # B x (N x (N-1)) x C_edge
        # 4. add HigherPri to edge attributes
        pri_mask = (batch_priorities.unsqueeze(2)>batch_priorities.unsqueeze(1)).to(torch.float32)
        higher_pri = torch.zeros(B, N, N-1).to(device)
        higher_pri[:, upper_pairing_idxs[0], upper_pairing_idxs[1]-1] = pri_mask[:, upper_pairing_idxs[0], upper_pairing_idxs[1]] 
        higher_pri[:, lower_pairing_idxs[0], lower_pairing_idxs[1]] = pri_mask[:, lower_pairing_idxs[0], lower_pairing_idxs[1]]
        higher_pri = higher_pri.view(B, -1)
        edge_attributes = torch.cat([edge_attributes, higher_pri.unsqueeze(-1)], dim=-1) # B x (N x (N-1)) x (C_edge+1)

        # Predict actions
        next_actions = self.gnn(node_concepts[0], edge_idxs, edge_attributes[0])

        return next_actions, node_concepts_explicit[0], edge_attributes[0]

class ResNetGNN(nn.Module):
    def __init__(self, config, mode):
        super(ResNetGNN, self).__init__()

        # Build feature extractor
        self.feature_extractor = LogicityFeatureExtractor()

        # Build node concept predictor
        ontology_file = config["ontology"]
        self.mode = mode
        self.read_ontology(ontology_file)

        node_pred_hidden = config["node_predictor"]["hidden"]
        self.node_channels = len(self.node_concept_names)*node_pred_hidden[2]

        self.node_concept_predictor = nn.Sequential(
            nn.Linear(self.feature_extractor.img_feature_channels, node_pred_hidden[0]),
            nn.BatchNorm1d(node_pred_hidden[0]),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(0.25),  # Dropout for regularization
            nn.Linear(node_pred_hidden[0], node_pred_hidden[1]),
            nn.BatchNorm1d(node_pred_hidden[1]),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(node_pred_hidden[1], self.node_channels),
        )
        self.node_concept_interpreter = nn.Sequential(
            nn.Linear(self.node_channels, len(self.node_concept_names)),
            nn.Sigmoid(),
        )

        # Build edge concept predictor
        # node attributes used for edge prediction: bbox + direction + priority
        self.bbox_channels = config["bbox_channels"]
        self.BBOX_POS_MAX = config["BBOX_POS_MAX"]
        self.edge_channels = len(self.edge_concept_names)
        edge_pred_hidden = config["node_predictor"]["hidden"]
        self.edge_predictor = nn.Sequential(
            nn.Linear(2*self.bbox_channels, edge_pred_hidden[0]),
            nn.ReLU(),
            nn.Linear(edge_pred_hidden[0], edge_pred_hidden[1]),
            nn.ReLU(),
            nn.Linear(edge_pred_hidden[1], self.edge_channels),
            nn.Sigmoid(),
        )

        # Reasoning engine
        gnn_config = config["gnn"]
        self.action_channels = len(self.action_names)
        self.edge_processor = nn.Sequential(
            nn.Linear(self.edge_channels, gnn_config["hidden"]),
            nn.ReLU(),
            nn.Linear(gnn_config["hidden"], self.node_channels*self.action_channels)
        ) # A neural network that maps edge features edge_attr of shape [-1, num_edge_features] to shape [-1, in_channels * out_channels]
        self.reasoning_engine = NNConv(in_channels=self.node_channels, out_channels=self.action_channels, nn=self.edge_processor) # don't support batch operation

    def read_ontology(self, ontology_file):

        assert self.mode in ["easy", "medium", "hard", "expert"]
        assert self.mode in ontology_file, "Ontology file does not contain mode: {}".format(self.mode)
        with open(ontology_file, 'r') as file:
            self.ontology_config = yaml.load(file, Loader=yaml.Loader)
        self.node_concept_names = []
        self.edge_concept_names = []
        self.action_names = ["Slow", "Fast", "Normal", "Stop"]
        for predicate in self.ontology_config["Predicates"]:
            predicate_name = list(predicate.keys())[0]
            if predicate[predicate_name]["arity"] == 1 and (predicate_name not in self.action_names): # "Is" assumtion is wrong, IsClose is binary
                self.node_concept_names.append(predicate_name)
            elif predicate[predicate_name]["arity"] == 2:
                self.edge_concept_names.append(predicate_name)
            else:
                assert predicate_name in self.action_names, "Unknown predicate name: {}".format(predicate_name)
            # additionally predict "Sees"
        self.edge_concept_names.append("Sees")

    def forward(self, batch_imgs, batch_bboxes, batch_directions, batch_priorities):
        roi_features = self.perceptor(batch_imgs, batch_bboxes)
        next_actions, unary_concepts, binary_concepts = \
            self.reasoning_engine(roi_features, batch_bboxes, batch_directions, batch_priorities)
        return next_actions, unary_concepts, binary_concepts

    def pred_concepts(self, batch_imgs, batch_bboxes, batch_directions, batch_priorities):
        roi_features = self.feature_extractor(batch_imgs, batch_bboxes)

        # concept prediction
        device = roi_features.device
        B = roi_features.shape[0]
        N = roi_features.shape[1]

        # Predict node concepts
        roi_features = roi_features.view(B*N, -1)
        node_concepts = self.node_concept_predictor(roi_features) # B x N x (concept_num x 256)
        # TODO: How to interpret the node_concepts?
        node_concepts_explicit = self.node_concept_interpreter(node_concepts)
        node_concepts = node_concepts.view(B, N, -1)

        # Create scene graph (node:(concept, bbox, direction, priority))
        # 1. concat node attributes use for edge prediction (bbox, direction)
        node_attributes = torch.cat([batch_bboxes/self.BBOX_POS_MAX, batch_directions, batch_priorities.unsqueeze(-1)], dim=-1) # B x N x (4+4+1)
        # 2. predict edge attributes
        node_attributes_paired = torch.zeros(B, N, N-1, 2*self.bbox_channels).to(device)
        upper_pairing_idxs = torch.triu_indices(N, N, offset=1).to(device)
        lower_pairing_idxs = torch.tril_indices(N, N, offset=-1).to(device)
        node_attributes_paired[:, upper_pairing_idxs[0], upper_pairing_idxs[1]-1] = torch.cat([
            node_attributes[:, upper_pairing_idxs[0]], node_attributes[:, upper_pairing_idxs[1]]
        ], dim=-1)
        node_attributes_paired[:, lower_pairing_idxs[0], lower_pairing_idxs[1]] = torch.cat([
            node_attributes[:, lower_pairing_idxs[0]], node_attributes[:, lower_pairing_idxs[1]]
        ], dim=-1)
        node_attributes_paired = node_attributes_paired.view(B, (N*(N-1)), -1)
        edge_attributes = self.edge_predictor(node_attributes_paired) # B x (N x (N-1)) x C_edge

        edge_attributes = edge_attributes.view(B, N, N-1, -1)

        return node_concepts, edge_attributes

    def reason(self, node_concepts, edge_attributes):
        N = node_concepts.shape[1]
        device = node_concepts.device

        tmp_idx = torch.arange(N)
        i, j = torch.meshgrid(tmp_idx, tmp_idx, indexing='ij')
        tmp_mask = (i != j)
        edge_idxs = torch.stack((i[tmp_mask],j[tmp_mask]),dim=1).T.to(device) # 2 x (N x (N-1))

        assert node_concepts.shape[0] == 1, "Batch operation not supported"
        edge_attributes = edge_attributes.view(1, N*(N-1), -1)
        next_actions = self.reasoning_engine(node_concepts[0], edge_idxs, edge_attributes[0])
        return next_actions