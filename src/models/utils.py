from src.models.backbone import ResNet18Backbone, SqueezeNet1Backbone, MobileNetV3Backbone
from src.models.baselinecnn import Conv1dNet
from src.models.microcnn import MicroCNN
from src.models.nanocnn import NanoCNN
from src.models.tinycnn import TinyCNN
from src.models.customcnn import CustomNet
from src.models.depwisecnn import MobileNetV1_1D, SimpleMobileNet1D
from src.models.resnet import  SmallResNet

model_map = {
    # "ResNet18Backbone": ResNet18Backbone,
    # "SqueezeNet1Backbone": SqueezeNet1Backbone,
    # "MobileNetV3Backbone": MobileNetV3Backbone,
    "MicroCNN": MicroCNN,
    "NanoCNN": NanoCNN,
    "TinyCNN": TinyCNN,
    "Conv1dNet": Conv1dNet,
    "CustomNet": CustomNet,
    "MobileNetV1_1D": MobileNetV1_1D,
    "SimpleMobileNet1D": SimpleMobileNet1D,
    "SmallResNet": SmallResNet
}