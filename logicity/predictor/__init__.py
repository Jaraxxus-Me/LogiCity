from .neural.ResNet_gnn import LogicityVisPredictorGNN
from .neural.ResNet_nlm import LogicityVisPredictorNLM


MODEL_BUILDER = {
    "LogicityVisPredictorGNN": LogicityVisPredictorGNN,
    "LogicityVisPredictorNLM": LogicityVisPredictorNLM,
}