import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Упрощенная версия без bottleneck
        self.conv1 = nn.Conv1d(in_features, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_features != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_features, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        # out = F.relu(out)

        return out


class SmallResNet(nn.Module):
    def __init__(self, num_classes=10, in_features=64):
        super(SmallResNet, self).__init__()

        # Начальные слои - уменьшаем каналы
        self.conv1 = nn.Conv1d(in_features, 8, kernel_size=5,
                               stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(8)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual blocks - уменьшаем количество блоков и каналов
        self.layer1 = self._make_layer(8, 8, 1, stride=1)  # 8->8
        self.layer2 = self._make_layer(8, 16, 1, stride=2)  # 8->16
        self.layer3 = self._make_layer(16, 32, 1, stride=2)  # 16->32

        # Финальные слои
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, n_mels, n_times)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
