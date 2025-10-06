import torch.nn as nn

class NanoCNN(nn.Module):
    def __init__(self, num_classes=2, in_features: int = 64):
        super(NanoCNN, self).__init__()

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
            nn.Conv1d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Глобальное усреднение
            nn.AdaptiveAvgPool1d(1)
        )

        # Минималистичный классификатор
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(48, num_classes)  # Прямо из 48 каналов в классы
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x