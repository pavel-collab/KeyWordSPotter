import torch.nn as nn

class MicroCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MicroCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Заменяем классификатор на адаптивный
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Этот слой преобразует любой вход в 1x1
            nn.Flatten(),
            nn.Linear(8, 16),  # Теперь на вход идет 8 каналов
            nn.ReLU(inplace=True),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x