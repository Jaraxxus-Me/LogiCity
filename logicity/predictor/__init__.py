from .neural.vis_predictor_gnn import LogicityVisPredictorGNN
from .neural.vis_predictor_nlm import ResNetNLM


MODEL_BUILDER = {
    "LogicityVisPredictorGNN": LogicityVisPredictorGNN,
    "ResNetNLM": ResNetNLM,
}