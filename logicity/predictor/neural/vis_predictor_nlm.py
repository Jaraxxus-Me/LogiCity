import torch
import yaml
import torch.nn as nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from logicity.rl_agent.policy.nlm_helper.nn.neural_logic import LogicMachine, LogitsInference
from logicity.predictor.neural.resnet_fpn import LogicityVisPerceptor

class NLM(nn.Module):
  """The model for family tree or general graphs path tasks."""

  def __init__(self, env, tgt_arity, nlm_args, \
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
        self.node_channels = len(self.node_concept_names)
        self.node_concept_predictor = nn.Sequential(
            nn.Linear(img_feature_channels, 512),
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
        self.nlm_args = {
            "input_dims": [0, len(self.node_concept_names), len(self.edge_concept_names), 0],
            "output_dims": 8,
            "logic_hidden_dim": [],
            "exclude_self": True,
            "depth": 4,
            "breadth": 3,
            "io_residual": False,
            "residual": False,
            "recursion": False,
        }
        self.nlm = NLM(env=None, tgt_arity=1, nlm_args=self.nlm_args, target_dim=self.action_channels)
    
    def forward(self, roi_features, batch_bboxes, batch_directions, batch_priorities):
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
            "n": N,
            "states": node_concepts, # B x N x C_node
            "relations": edge_attributes_.view(B, N, N, -1), # B x N x N x (C_edge+1)
        }
        next_actions = self.nlm(feed_dict)[0]  # TODO: don't support batch now

        return next_actions, node_concepts[0], edge_attributes[0].view(N*(N-1), -1)        


class LogicityVisPredictorNLM(nn.Module):
    def __init__(self, mode):
        super(LogicityVisPredictorNLM, self).__init__()

        self.perceptor = LogicityVisPerceptor()
        self.reasoning_engine = LogicityVisReasoningEngine(mode, self.perceptor.img_feature_channels)

    def forward(self, batch_imgs, batch_bboxes, batch_directions, batch_priorities):
        roi_features = self.perceptor(batch_imgs, batch_bboxes)
        next_actions, unary_concepts, binary_concepts = \
            self.reasoning_engine(roi_features, batch_bboxes, batch_directions, batch_priorities)
        return next_actions, unary_concepts, binary_concepts