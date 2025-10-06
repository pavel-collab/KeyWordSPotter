import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv1D, self).__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=3,
            stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm1d(in_channels)

        # Pointwise convolution
        self.pointwise = nn.Conv1d(
            in_channels, out_channels, kernel_size=1,
            stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x


class MobileNetV1_1D(nn.Module):
    def __init__(self, num_classes=10, alpha=0.2, in_features=64):
        super(MobileNetV1_1D, self).__init__()

        # Первый обычный сверточный слой
        first_channels = max(10, int(20 * alpha))
        self.features = nn.Sequential(
            nn.Conv1d(in_features, first_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(first_channels),
            nn.ReLU(inplace=True)
        )

        config = [
            (32, 1),
            (64, 2),
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
        ]

        # Применяем alpha ко всем каналам
        config = [(max(10, int(out_channels * alpha)), stride) for out_channels, stride in config]

        # Создаем последовательность блоков
        in_channels = first_channels
        for i, (out_channels, stride) in enumerate(config):
            self.features.add_module(
                f'dw_conv_{i}',
                DepthwiseSeparableConv1D(in_channels, out_channels, stride)
            )
            in_channels = out_channels

        # Global Average Pooling и классификатор
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 24),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(24, num_classes),
        )

        # Инициализация весов
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class SimpleMobileNet1D(nn.Module):
    def __init__(self, num_classes=10, alpha=0.15, in_features=64):
        super(SimpleMobileNet1D, self).__init__()

        first_channels = max(8, int(32 * alpha))
        self.features = nn.Sequential(
            nn.Conv1d(in_features, first_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(first_channels),
            nn.ReLU(inplace=True),

            # Слегка уменьшены каналы в блоках
            DepthwiseSeparableConv1D(first_channels, max(14, int(56 * alpha)), stride=1),  # было 16
            DepthwiseSeparableConv1D(max(14, int(56 * alpha)), max(28, int(112 * alpha)), stride=2),  # было 32
            DepthwiseSeparableConv1D(max(28, int(112 * alpha)), max(42, int(224 * alpha)), stride=2),  # было 48
            DepthwiseSeparableConv1D(max(42, int(224 * alpha)), max(56, int(448 * alpha)), stride=2),  # было 64
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(max(56, int(448 * alpha)), 28),  # уменьшено с 32 до 28
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(28, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x