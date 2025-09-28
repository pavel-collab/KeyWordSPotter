import torch
import torch.nn as nn

class Conv1dNet(torch.nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            # Первый сверточный блок
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(16),
            torch.nn.MaxPool2d(kernel_size=2),
            
            # Второй сверточный блок
            torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(16),
            torch.nn.MaxPool2d(kernel_size=2),
            
            # Третий сверточный блок
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(32),
            torch.nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(16, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x