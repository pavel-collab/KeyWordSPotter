import torch
import torch.nn as nn
import torchvision.models as models

import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

class ResNet18Backbone(nn.Module):
    def __init__(self, num_classes: int = 2, in_features: int = 1):
        super().__init__()
                
        # Загружаем предобученную модель
        # self.backbone = getattr(models, backbone)(pretrained=True)
        #! parameter pretrained is depricated now, it's recommended to use weights instead
        self.backbone = getattr(models, "resnet18")(weights='DEFAULT')
        
        # Адаптируем первый слой для аудио (1 входной канал вместо 3)
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        
        # Копируем веса для первого канала (усредняем RGB веса)
        with torch.no_grad():
            # Для 1 канала: усредняем веса по RGB каналам
            new_weight = original_conv1.weight.mean(dim=1, keepdim=True)
            self.backbone.conv1.weight.copy_(new_weight)
        
        # Заменяем последний слой
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = x[:,None,:,:]
        return self.backbone(x)

class SqueezeNet1Backbone(nn.Module):
    def __init__(self, num_classes: int = 2, in_features: int = 1):
        super().__init__()
        
        # Load the pre-trained model
        self.backbone = getattr(models, "squeezenet1_0")(weights='DEFAULT')
        
        # Adapt the first convolutional layer for audio (1 input channel instead of 3)
        # In SqueezeNet, the first layer is 'features.conv1'
        original_conv1 = self.backbone.features[0]
        self.backbone.features[0] = nn.Conv2d(
            in_channels=1, 
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=(original_conv1.bias is not None)
        )
        
        # Copy the weights for the first channel (average the RGB weights)
        with torch.no_grad():
            new_weight = original_conv1.weight.mean(dim=1, keepdim=True)
            self.backbone.features[0].weight.copy_(new_weight)
            
            # If you need to copy bias
            if original_conv1.bias is not None:
                self.backbone.features[0].bias.copy_(original_conv1.bias)
        
        # Replace the final classifier
        # In SqueezeNet, the final classifier is 'classifier'
        original_conv_final = self.backbone.classifier[1]
        self.backbone.classifier[1] = nn.Conv2d(
            original_conv_final.in_channels, 
            num_classes,
            kernel_size=original_conv_final.kernel_size,
            stride=original_conv_final.stride,
            padding=original_conv_final.padding,
            bias=(original_conv_final.bias is not None)
        )

    def forward(self, x):
        x = x[:,None,:,:]
        return self.backbone(x)

class MobileNetV3Backbone(nn.Module):
    def __init__(self, num_classes: int = 2, in_features: int = 1):
        super().__init__()
        
        # Загружаем предобученную модель
        self.backbone = getattr(models, "mobilenet_v3_small")(weights='DEFAULT')
        
        # Адаптируем первый слой для аудио (1 входной канал вместо 3)
        # В MobileNetV3 первый слой называется 'features[0].0' (Conv2d)
        original_conv1 = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=1, 
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )
        
        # Копируем веса для первого канала (усредняем RGB веса)
        with torch.no_grad():
            # Для 1 канала: усредняем веса по RGB каналам
            new_weight = original_conv1.weight.mean(dim=1, keepdim=True)
            self.backbone.features[0][0].weight.copy_(new_weight)
            # Копируем bias, если он есть
            if original_conv1.bias is not None:
                self.backbone.features[0][0].bias.copy_(original_conv1.bias)
        
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
        x = x[:,None,:,:]
        return self.backbone(x)