import torch
import yaml
import torch.nn as nn
from logicity.rl_agent.policy.nlm_helper.nn.neural_logic import LogicMachine, LogitsInference
from logicity.predictor.neural.resnet_fpn import LogicityFeatureExtractor

class NLM(nn.Module):
  """The model for family tree or general graphs path tasks."""

  def __init__(self, tgt_arity, nlm_args, \
               target_dim):
    super().__init__()
    # inputs
    self.feature_axis = tgt_arity
    self.nlm_args = nlm_args
    self.features = LogicMachine(**nlm_args)
    output_dim = self.features.output_dims[self.feature_axis]
    # Do not sigmoid as we will use CrossEntropyLoss
    self.pred = LogitsInference(output_dim, target_dim, [])

  def forward(self, feed_dict):
    states = feed_dict['states']
    relations = feed_dict['relations']
    inp = [None for _ in range(self.nlm_args['breadth'] + 1)]
    inp[1] = states
    inp[2] = relations
    depth = None
    feature = self.features(inp, depth=depth)[self.feature_axis]
    pred = self.pred(feature)
    return pred


def get_nlm_binary_concepts(batch_relation_matrix, batch_edge_index):
    B, N, N, R = batch_relation_matrix.shape
    gt_binary_concepts = torch.zeros((B, N, N, N, R)).to(batch_relation_matrix.device)

    for b in range(len(batch_edge_index)):
        edge_index = batch_edge_index[b]
        edge_dict = edge_index2dict(edge_index, N)
        for n in range(N):
            see_ent = edge_dict[n]
            for i, i_ in enumerate(see_ent):
                for j, j_ in enumerate(see_ent):
                    gt_binary_concepts[b, n, i, j] = batch_relation_matrix[b, i_, j_]

    return gt_binary_concepts.reshape(B*N, N, N, R)

def get_nlm_unary_concepts(batch_unary_concepts, batch_edge_index):
    B, N, U = batch_unary_concepts.shape
    gt_unary_concepts = torch.zeros((B, N, N, U)).to(batch_unary_concepts.device)

    for b in range(len(batch_edge_index)):
        edge_index = batch_edge_index[b]
        edge_dict = edge_index2dict(edge_index, N)
        for n in range(N):
            see_ent = edge_dict[n]
            gt_unary_concepts[b, n, :len(see_ent)] = batch_unary_concepts[b, see_ent]

    return gt_unary_concepts.reshape(B*N, N, U)

def edge_index2dict(edge_index, num_nodes):
    edge_dict = {}
    for i, (src, dst) in enumerate(zip(edge_index[0], edge_index[1])):
        if dst.item() not in edge_dict:
            edge_dict[dst.item()] = [dst.item()]
        edge_dict[dst.item()].append(src.item())
    for n in range(num_nodes):
        if n not in edge_dict:
            edge_dict[n] = [n]
    return edge_dict

class ResNetNLM(nn.Module):
    def __init__(self, config, mode):
        super(ResNetNLM, self).__init__()
        
        # Build feature extractor
        self.grounding_net = GroundingNet(config, mode)

        # Reasoning engine
        nlm_args = config["nlm_args"]
        nlm_args["input_dims"] = [0, len(self.grounding_net.node_concept_names), \
                                  len(self.grounding_net.edge_concept_names), 0]
        self.reasoning_net = ReasoningNet(self.grounding_net.action_names, config["nlm_args"], \
                                                   config["nlm_arity"])


    def forward(self, batch_imgs, batch_bboxes, batch_directions, batch_priorities, batch_edge_index):
        node_concepts, edge_concepts = self.grounding_net(batch_imgs, batch_bboxes, batch_directions, \
                                                          batch_priorities, batch_edge_index)
        feed_dict = {
            "n": torch.tensor([node_concepts.shape[1]]*node_concepts.shape[0]), # [5] * (B*N)
            "states": node_concepts, # BN x 5 x C_node
            "relations": edge_concepts, # BN x 5 x 5 x (C_edge+1)
        }
        next_actions = self.reasoning_net(feed_dict)
        return next_actions
    
    def grounding(self, batch_imgs, batch_bboxes, batch_directions, batch_priorities, batch_edge_index):
        node_concepts, edge_concepts = self.grounding_net(batch_imgs, batch_bboxes, batch_directions, batch_priorities, batch_edge_index)
        return node_concepts, edge_concepts
    
    def reasoning(self, feed_dict):
        next_actions = self.reasoning_net(feed_dict)
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

    def get_node_concepts(self, batch_imgs, batch_bboxes, batch_edge_index):
        roi_features = self.feature_extractor(batch_imgs, batch_bboxes)
        B = roi_features.shape[0]
        N = roi_features.shape[1]

        roi_features = roi_features.view(B*N, -1)
        
        node_features = self.node_predictor(roi_features).reshape(B, N, -1)
        node_concepts = self.node_concept_interpreter(node_features) # B x N x C_node

        node_concepts = get_nlm_unary_concepts(node_concepts, batch_edge_index)

        return node_concepts
    
    def get_edge_concepts(self, batch_bboxes, batch_directions, batch_priorities, batch_edge_index):
        B, N = batch_bboxes.shape[:2]
        node_attributes = torch.cat([batch_bboxes/self.BBOX_POS_MAX, batch_directions, batch_priorities.unsqueeze(-1)], dim=-1) # B x N x (4+4+1)
        # 2. predict edge attributes
        node_attributes_paired = torch.zeros(B, N, N, 2*self.bbox_channels).to(node_attributes.device)
        # pair node attributes
        node_attr_expanded = node_attributes.unsqueeze(2).expand(B, N, N, self.bbox_channels)
        node_attr_tiled = node_attributes.unsqueeze(1).expand(B, N, N, self.bbox_channels)

        # Concatenate along the last dimension to pair the attributes
        node_attributes_paired = torch.cat([node_attr_expanded, node_attr_tiled], dim=-1)
        node_attributes_paired = node_attributes_paired.view(-1, 2*self.bbox_channels)
        edge_features = self.edge_predictor(node_attributes_paired) # B x (N x N) x C_edge
        edge_concepts = self.edge_concept_interpreter(edge_features).reshape(B, N, N, -1)

        edge_concepts = get_nlm_binary_concepts(edge_concepts, batch_edge_index)

        return edge_concepts
    
    def forward(self, batch_imgs, batch_bboxes, batch_directions, batch_priorities, batch_edge_index):
        node_concepts = self.get_node_concepts(batch_imgs, batch_bboxes, batch_edge_index)
        edge_concepts = self.get_edge_concepts(batch_bboxes, batch_directions, batch_priorities, batch_edge_index)

        return node_concepts, edge_concepts
    
    
class ReasoningNet(nn.Module):
    def __init__(self, action_names, nlm_args, target_dim):
        super(ReasoningNet, self).__init__()

        # Build action predictor
        self.action_channels = len(action_names)
        self.nlm_args = nlm_args
        self.nlm = NLM(tgt_arity=target_dim, nlm_args=self.nlm_args, target_dim=self.action_channels)
    
    def forward(self, feed_dict):
        next_actions = self.nlm(feed_dict)
        # take ego agent action
        next_actions = next_actions[:, 0, :]
        return next_actions