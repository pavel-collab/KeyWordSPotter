from src.models.backbone import ResNetBackbone, SqueezeNet1Backbone, MobileNetV3Backbone
from src.models.baselinecnn import Conv1dNet
from src.models.microcnn import MicroCNN
from src.models.nanocnn import NanoCNN
from src.models.tinycnn import TinyCNN

model_map = {
    "resnet8": ResNetBackbone,
    "resnet18": ResNetBackbone,
    "squeezenet1_0": SqueezeNet1Backbone,
    "mobilenet_v3_small": MobileNetV3Backbone,
    "micro_cnn": MicroCNN,
    "nano_cnn": NanoCNN,
    "tiny_cnn": TinyCNN,
    "baseline": Conv1dNet
}