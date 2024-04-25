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

class NLMReasoningEngine(nn.Module):
    def __init__(self, action_names, nlm_args, target_dim):
        super(NLMReasoningEngine, self).__init__()

        # Build action predictor
        self.action_channels = len(action_names)
        self.nlm_args = nlm_args
        self.nlm = NLM(tgt_arity=target_dim, nlm_args=self.nlm_args, target_dim=self.action_channels)
    
    def forward(self, feed_dict):
        next_actions = self.nlm(feed_dict)
        return next_actions


class ResNetNLM(nn.Module):
    def __init__(self, config, mode):
        super(ResNetNLM, self).__init__()
        
        # Build feature extractor
        self.feature_extractor = LogicityFeatureExtractor()

        # Build node concept predictor
        ontology_file = config["ontology"]
        self.mode = mode
        self.read_ontology(ontology_file)

        self.node_channels = len(self.node_concept_names)
        node_pred_hidden = config["node_predictor"]["hidden"]

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
        nlm_args = config["nlm_args"]
        nlm_args["input_dims"] = [0, len(self.node_concept_names), len(self.edge_concept_names), 0]
        self.reasoning_engine = NLMReasoningEngine(self.action_names, config["nlm_args"], \
                                                   config["nlm_arity"])


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
        node_concepts, edge_attributes = self.pred_concepts(batch_imgs, batch_bboxes, batch_directions, batch_priorities)
        next_actions = self.reason(node_concepts, edge_attributes)
        return next_actions, node_concepts, edge_attributes

    def pred_concepts(self, batch_imgs, batch_bboxes, batch_directions, batch_priorities):
        roi_features = self.feature_extractor(batch_imgs, batch_bboxes)

        # concept prediction
        device = roi_features.device
        B = roi_features.shape[0]
        N = roi_features.shape[1]

        # Predict node concepts
        node_concepts = self.node_concept_predictor(roi_features.view(-1, self.feature_extractor.img_feature_channels)) # B x N x concept_num
        node_concepts = node_concepts.view(B, N, -1)
        
        # Create scene graph (node:(concept, bbox, direction, priority))
        # 1. concat node attributes use for edge prediction (bbox, direction)
        node_attributes = torch.cat([batch_bboxes/self.BBOX_POS_MAX, batch_directions, batch_priorities.unsqueeze(-1)], dim=-1) # B x N x (4+4+1)
        # 2. predict edge attributes
        node_attributes_paired = torch.zeros(B, N, N, 2*self.bbox_channels).to(device)
        # pair node attributes
        node_attr_expanded = node_attributes.unsqueeze(2).expand(B, N, N, self.bbox_channels)
        node_attr_tiled = node_attributes.unsqueeze(1).expand(B, N, N, self.bbox_channels)

        # Concatenate along the last dimension to pair the attributes
        node_attributes_paired = torch.cat([node_attr_expanded, node_attr_tiled], dim=-1)
        node_attributes_paired = node_attributes_paired.view(-1, 2*self.bbox_channels)
        edge_attributes = self.edge_predictor(node_attributes_paired) # B x (N x N) x C_edge
        edge_attributes = edge_attributes.view(B, N, N, -1)

        return node_concepts, edge_attributes
    
    def reason(self, node_concepts, edge_attributes):
        # Predict actions
        feed_dict = {
            "n": torch.tensor([node_concepts.shape[1]]*node_concepts.shape[0]),
            "states": node_concepts, # B x N x C_node
            "relations": edge_attributes, # B x N x N x (C_edge+1)
        }
        next_actions = self.reasoning_engine(feed_dict)

        return next_actions