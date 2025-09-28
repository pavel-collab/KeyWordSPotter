import torch.nn as nn

class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(TinyCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Заменяем классификатор на адаптивный
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Автоматически преобразует любой вход в 1x1
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(16, 32),  # Теперь на вход идет 16 каналов (выход последнего Conv2d)
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x