import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Backbone(nn.Module):
    def __init__(self, num_classes: int = 2, in_features: int = 64):
        super().__init__()
                
        # Загружаем предобученную модель
        # self.backbone = getattr(models, backbone)(pretrained=True)
        #! parameter pretrained is depricated now, it's recommended to use weights instead
        self.backbone = getattr(models, "resnet18")(weights='DEFAULT')
        
        # Заменяем последний слой
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

class SqueezeNet1Backbone(nn.Module):
    def __init__(self, num_classes: int = 2, in_features: int = 64):
        super().__init__()
        
        # Load the pre-trained model
        self.backbone = getattr(models, "squeezenet1_0")(weights='DEFAULT')
        
        # Replace the final classifier
        # In SqueezeNet, the final classifier is 'classifier'
        original_conv_final = self.backbone.classifier[1]
        self.backbone.classifier[1] = nn.Conv1d(
            original_conv_final.in_channels, 
            num_classes,
            kernel_size=original_conv_final.kernel_size,
            stride=original_conv_final.stride,
            padding=original_conv_final.padding,
            bias=(original_conv_final.bias is not None)
        )

    def forward(self, x):
        return self.backbone(x)

class MobileNetV3Backbone(nn.Module):
    def __init__(self, num_classes: int = 2, in_features: int = 64):
        super().__init__()
        
        # Загружаем предобученную модель
        self.backbone = getattr(models, "mobilenet_v3_small")(weights='DEFAULT')
        
        # Заменяем последний классификатор
        # В MobileNetV3 последний слой называется 'classifier'
        # Сохраняем исходное количество входных特征
        original_classifier = self.backbone.classifier
        in_features = original_classifier[-1].in_features
        
        # Создаем новый классификатор (последний Linear слой)
        self.backbone.classifier = nn.Sequential(
            *list(original_classifier.children())[:-1],  # Сохраняем все слои кроме последнего
            nn.Linear(in_features, num_classes)  # Заменяем последний слой
        )

    def forward(self, x):
        return self.backbone(x)