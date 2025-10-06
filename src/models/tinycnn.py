import torch.nn as nn


class TinyCNN(nn.Module):
    def __init__(self, num_classes=10, in_features: int = 64):
        super(TinyCNN, self).__init__()
        self.features = nn.Sequential(
            # Первый блок
            nn.Conv1d(in_features, 14, kernel_size=3, padding=1),
            nn.BatchNorm1d(14),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Второй блок
            nn.Conv1d(14, 28, kernel_size=3, padding=1),
            nn.BatchNorm1d(28),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Третий блок
            nn.Conv1d(28, 42, kernel_size=3, padding=1),
            nn.BatchNorm1d(42),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(42, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x