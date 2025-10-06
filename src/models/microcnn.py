import torch.nn as nn


class MicroCNN(nn.Module):
    def __init__(self, num_classes=10, in_features=64):
        super(MicroCNN, self).__init__()

        self.num_classes = num_classes

        self.features = nn.Sequential(
            # Первый блок
            nn.Conv1d(in_features, 12, kernel_size=3, padding=1),
            nn.BatchNorm1d(12),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Второй блок
            nn.Conv1d(12, 24, kernel_size=3, padding=1),
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Третий блок
            nn.Conv1d(24, 36, kernel_size=3, padding=1),
            nn.BatchNorm1d(36),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(36, 24),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(24, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x