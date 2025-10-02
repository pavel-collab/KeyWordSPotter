import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution (отдельная свертка для каждого канала)
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, 
            stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution (1×1 свертка для комбинирования каналов)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, 
            stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        return x
    
class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0):
        """
        Args:
            num_classes: количество классов для классификации
            alpha: коэффициент ширины сети (0.25, 0.5, 0.75, 1.0)
        """
        super(MobileNetV1, self).__init__()
        
        # Первый обычный сверточный слой
        first_channels = int(32 * alpha)
        self.features = nn.Sequential(
            nn.Conv2d(3, first_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(first_channels),
            nn.ReLU(inplace=True)
        )
        
        # Конфигурация depthwise separable блоков
        # (out_channels, stride)
        config = [
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (512, 2),
            (512, 1), (512, 1), (512, 1), (512, 1), (512, 1),  # 6 повторов
            (1024, 2),
            (1024, 1)
        ]
        
        # Применяем alpha ко всем каналам
        config = [(int(out_channels * alpha), stride) for out_channels, stride in config]
        
        # Создаем последовательность блоков
        in_channels = first_channels
        for i, (out_channels, stride) in enumerate(config):
            self.features.add_module(
                f'dw_conv_{i}',
                DepthwiseSeparableConv(in_channels, out_channels, stride)
            )
            in_channels = out_channels
        
        # Global Average Pooling и классификатор
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, num_classes)
        
        # Инициализация весов
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
class SimpleMobileNet(nn.Module):
    def __init__(self, num_classes=10, alpha=1.0):
        super(SimpleMobileNet, self).__init__()
        
        first_channels = int(32 * alpha)
        self.features = nn.Sequential(
            # Первый слой для маленьких изображений (32x32)
            nn.Conv2d(3, first_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(first_channels),
            nn.ReLU(inplace=True),
            
            # Упрощенная конфигурация блоков
            DepthwiseSeparableConv(first_channels, int(64 * alpha), stride=1),
            DepthwiseSeparableConv(int(64 * alpha), int(128 * alpha), stride=2),
            DepthwiseSeparableConv(int(128 * alpha), int(256 * alpha), stride=2),
            DepthwiseSeparableConv(int(256 * alpha), int(512 * alpha), stride=2),
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(int(512 * alpha), num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x