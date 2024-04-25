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
            nn.ReLU(),
            nn.Linear(node_pred_hidden[0], node_pred_hidden[1]),
            nn.ReLU(),
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
        roi_features = self.feature_extractor(batch_imgs, batch_bboxes)

        # concept prediction
        device = roi_features.device
        B = roi_features.shape[0]
        N = roi_features.shape[1]

        # Predict node concepts
        node_concepts = self.node_concept_predictor(roi_features) # B x N x concept_num
        
        # Create scene graph (node:(concept, bbox, direction, priority))
        # 1. concat node attributes use for edge prediction (bbox, direction)
        node_attributes = torch.cat([batch_bboxes/self.BBOX_POS_MAX, batch_directions], dim=-1) # B x N x (4+4)
        # 2. predict edge attributes
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
        # 3. add HigherPri to edge attributes
        pri_mask = (batch_priorities.unsqueeze(2)>batch_priorities.unsqueeze(1)).to(torch.float32)
        higher_pri = torch.zeros(B, N, N-1).to(device)
        higher_pri[:, upper_pairing_idxs[0], upper_pairing_idxs[1]-1] = pri_mask[:, upper_pairing_idxs[0], upper_pairing_idxs[1]] 
        higher_pri[:, lower_pairing_idxs[0], lower_pairing_idxs[1]] = pri_mask[:, lower_pairing_idxs[0], lower_pairing_idxs[1]]
        higher_pri = higher_pri.view(B, -1)
        edge_attributes = torch.cat([edge_attributes, higher_pri.unsqueeze(-1)], dim=-1) # B x (N x (N-1)) x (C_edge+1)
        edge_attributes = edge_attributes.view(B, N, N-1, -1) # B x N x (N-1) x (C_edge+1)

        # Predict actions
        edge_attributes_ = torch.zeros(B, N, N, edge_attributes.shape[-1]).to(device)
        edge_attributes_[:, upper_pairing_idxs[0], upper_pairing_idxs[1]] = \
            edge_attributes[:, upper_pairing_idxs[0], upper_pairing_idxs[1]-1]
        edge_attributes_[:, lower_pairing_idxs[0], lower_pairing_idxs[1]] = \
            edge_attributes[:, lower_pairing_idxs[0], lower_pairing_idxs[1]]
        feed_dict = {
            "n": torch.tensor([N]*B),
            "states": node_concepts, # B x N x C_node
            "relations": edge_attributes_.view(B, N, N, -1), # B x N x N x (C_edge+1)
        }
        next_actions = self.reasoning_engine(feed_dict)

        return next_actions, node_concepts, edge_attributes
