import torch.nn as nn

class NanoCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(NanoCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(2, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Заменяем классификатор на адаптивный
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Автоматически преобразует любой вход в 1x1
            nn.Flatten(),
            nn.Linear(4, 8),  # Теперь на вход идет 4 канала (выход последнего Conv2d)
            nn.ReLU(inplace=True),
            nn.Linear(8, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x