import torch
import yaml
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv
from logicity.predictor.neural.resnet_fpn import LogicityFeatureExtractor


class ResNetGNN(nn.Module):
    def __init__(self, config, mode):
        super(ResNetGNN, self).__init__()
        # Grounding engine 
        self.grounding_net = GroundingNet(config, mode)
        # Reasoning engine
        self.reasoning_net = ReasoningNet(config["gnn"], self.grounding_net.node_channels, self.grounding_net.edge_channels, self.grounding_net.action_channels)
    
    def create_batch_graph(self, node_concepts, edge_concepts, edge_indices):
        data_list = []
        num_graphs = node_concepts.size(0)
        for i in range(num_graphs):
            graph_data = Data(x=node_concepts[i], edge_index=edge_indices[i], edge_attr=edge_concepts[i])
            data_list.append(graph_data)

        batch_graph = Batch.from_data_list(data_list)
        return batch_graph
    
    def create_sliced_edge_concepts(self, edge_concepts, batch_edge_index):
        edge_num_list = [edge_index.shape[1] for edge_index in batch_edge_index]
        start_indices = [sum(edge_num_list[:i]) for i in range(len(edge_num_list))]
        sliced_edge_concepts = [edge_concepts[start:start+size] for start, size in zip(start_indices, edge_num_list)]
        return sliced_edge_concepts
    
    def forward(self, batch_imgs, batch_bboxes, batch_directions, batch_priorities, batch_edge_index):
        node_concepts, edge_concepts = self.grounding_net(batch_imgs, batch_bboxes, batch_directions, batch_priorities, batch_edge_index)
        sliced_edge_concepts = self.create_sliced_edge_concepts(edge_concepts, batch_edge_index)
        batch_graph = self.create_batch_graph(node_concepts, sliced_edge_concepts, batch_edge_index) 
        next_actions = self.reasoning_net(batch_graph)        
        return next_actions
    
    def grounding(self, batch_imgs, batch_bboxes, batch_directions, batch_priorities, batch_edge_index):
        node_concepts, edge_concepts = self.grounding_net(batch_imgs, batch_bboxes, batch_directions, batch_priorities, batch_edge_index)
        return node_concepts, edge_concepts
    
    def reasoning(self, batch_graph):
        next_actions = self.reasoning_net(batch_graph)
        return next_actions


class GroundingNet(nn.Module):
    def __init__(self, config, mode):
        super(GroundingNet, self).__init__()
        ontology_file = config["ontology"]
        self.mode = mode
        self.read_ontology(ontology_file)

        self.node_channels = len(self.node_concept_names) # 12
        self.edge_channels = len(self.edge_concept_names) # 3
        self.action_channels = len(self.action_names) # 4

        self.feature_extractor = LogicityFeatureExtractor()
        node_pred_hidden = config["node_predictor"]["hidden"]

        self.node_predictor = nn.Sequential(
            nn.Linear(self.feature_extractor.img_feature_channels, node_pred_hidden[0]),
            nn.BatchNorm1d(node_pred_hidden[0]),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(0.25),  # Dropout for regularization
            nn.Linear(node_pred_hidden[0], node_pred_hidden[1]),
            nn.BatchNorm1d(node_pred_hidden[1]),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(node_pred_hidden[1], node_pred_hidden[2]),
            nn.ReLU(),
        )
        self.node_concept_interpreter = nn.Sequential(
            nn.Linear(node_pred_hidden[2], self.node_channels),
            nn.Sigmoid(),
        )

        self.bbox_channels = config["bbox_channels"]
        self.BBOX_POS_MAX = config["BBOX_POS_MAX"]
        
        edge_pred_hidden = config["edge_predictor"]["hidden"]
        self.edge_predictor = nn.Sequential(
            nn.Linear(2*self.bbox_channels, edge_pred_hidden[0]),
            nn.ReLU(),
            nn.Linear(edge_pred_hidden[0], edge_pred_hidden[1]),
            nn.ReLU(),
        )

        self.edge_concept_interpreter = nn.Sequential(
            nn.Linear(edge_pred_hidden[1], self.edge_channels),
            nn.Sigmoid(),
        )

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

    def get_node_concepts(self, batch_imgs, batch_bboxes):
        roi_features = self.feature_extractor(batch_imgs, batch_bboxes)
        B = roi_features.shape[0]
        N = roi_features.shape[1]

        roi_features = roi_features.view(B*N, -1)
        
        node_features = self.node_predictor(roi_features).reshape(B, N, -1)
        node_concepts = self.node_concept_interpreter(node_features) # B x N x C_node

        return node_concepts
    
    def get_edge_concepts(self, batch_bboxes, batch_directions, batch_priorities, batch_edge_index):
        node_attributes = torch.cat([batch_bboxes/self.BBOX_POS_MAX, batch_directions, batch_priorities.unsqueeze(-1)], dim=-1) # B x N x (4+4+1)

        src_nodes_list = []
        dst_nodes_list = []
        for i, edge_index in enumerate(batch_edge_index):
            src_nodes = edge_index[0]
            src_nodes_list.append(node_attributes[i][src_nodes])
            dst_nodes = edge_index[1]
            dst_nodes_list.append(node_attributes[i][dst_nodes])
            
        src_node_attr = torch.cat(src_nodes_list, dim=0)
        dst_node_attr = torch.cat(dst_nodes_list, dim=0)

        edge_attributes = torch.cat((src_node_attr, dst_node_attr), dim=1) # (B * |E|) x 18
        edge_features = self.edge_predictor(edge_attributes)
        edge_concepts = self.edge_concept_interpreter(edge_features) # (B * |E|) x C_edge

        return edge_concepts
    
    def forward(self, batch_imgs, batch_bboxes, batch_directions, batch_priorities, batch_edge_index):
        node_concepts = self.get_node_concepts(batch_imgs, batch_bboxes)
        edge_concepts = self.get_edge_concepts(batch_bboxes, batch_directions, batch_priorities, batch_edge_index)

        return node_concepts, edge_concepts


class ReasoningNet(nn.Module):
    def __init__(self, gnn_config, node_channels, edge_channels, action_channels):
        super(ReasoningNet, self).__init__()
        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.action_channels = action_channels

        self.conv1 = GINEConv(nn=nn.Sequential(
            nn.Linear(self.node_channels, gnn_config["hidden"]),
            nn.ReLU(),
            nn.Linear(gnn_config["hidden"], gnn_config["hidden"]),
            nn.ReLU()), 
            edge_dim=self.edge_channels)

        self.conv2 = GINEConv(nn=nn.Sequential(
            nn.Linear(gnn_config["hidden"], gnn_config["hidden"]),
            nn.ReLU(),
            nn.Linear(gnn_config["hidden"], gnn_config["hidden"]),
            nn.ReLU()), 
            edge_dim=self.edge_channels)

        self.pred_layer = nn.Linear(gnn_config["hidden"], self.action_channels)

    def forward(self, batch_graph):
        x, edge_index, edge_attr = batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.pred_layer(x)
        return x