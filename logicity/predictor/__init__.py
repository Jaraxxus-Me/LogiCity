from .neural.vis_predictor_gnn import LogicityVisPredictorGNN
from .neural.vis_predictor_nlm import LogicityVisPredictorNLM


MODEL_BUILDER = {
    "LogicityVisPredictorGNN": LogicityVisPredictorGNN,
    "LogicityVisPredictorNLM": LogicityVisPredictorNLM,
}