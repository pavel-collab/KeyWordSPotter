from src.models.backbone import ResNet18Backbone, SqueezeNet1Backbone, MobileNetV3Backbone
from src.models.baselinecnn import Conv1dNet
from src.models.microcnn import MicroCNN
from src.models.nanocnn import NanoCNN
from src.models.tinycnn import TinyCNN

model_map = {
    # "resnet18": ResNet18Backbone,
    # "squeezenet1_0": SqueezeNet1Backbone,
    # "mobilenet_v3_small": MobileNetV3Backbone,
    "MicroCNN": MicroCNN,
    "NanoCNN": NanoCNN,
    "TinyCNN": TinyCNN,
    "Conv1dNet": Conv1dNet
}