from .neural.vis_predictor_gnn import ResNetGNN
from .neural.vis_predictor_nlm import ResNetNLM


MODEL_BUILDER = {
    "ResNetGNN": ResNetGNN,
    "ResNetNLM": ResNetNLM,
}