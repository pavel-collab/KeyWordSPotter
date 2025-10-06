import torch
import torch.nn as nn


class Conv1dNet(torch.nn.Module):
    def __init__(self, num_classes=5, in_features=64):
        super().__init__()
        self.features = nn.Sequential(
            # Первый блок - используем меньшие каналы
            torch.nn.Conv1d(in_channels=in_features, out_channels=20, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(num_features=20),
            torch.nn.MaxPool1d(kernel_size=2),

            # Второй блок
            torch.nn.Conv1d(in_channels=20, out_channels=28, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(num_features=28),
            torch.nn.MaxPool1d(kernel_size=2),

            # Третий блок
            torch.nn.Conv1d(in_channels=28, out_channels=36, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(num_features=36),
        )

        self.classifier = nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(36, 18),
            nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(18, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x