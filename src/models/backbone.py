import torch
import torch.nn as nn
import torchvision.models as models

class CNNBackbone(nn.Module):
    def __init__(self, backbone: str = "resnet18", num_classes: int = 2, in_channels: int = 1):
        super().__init__()
                
        # Загружаем предобученную модель
        # self.backbone = getattr(models, backbone)(pretrained=True)
        #! parameter pretrained is depricated now, it's recommended to use weights instead
        self.backbone = getattr(models, backbone)(weights='DEFAULT')
        
        # Адаптируем первый слой для аудио (1 входной канал вместо 3)
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        
        # Копируем веса для первого канала (усредняем RGB веса)
        with torch.no_grad():
            if in_channels == 1:
                # Для 1 канала: усредняем веса по RGB каналам
                new_weight = original_conv1.weight.mean(dim=1, keepdim=True)
                self.backbone.conv1.weight.copy_(new_weight)
        
        # Заменяем последний слой
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)