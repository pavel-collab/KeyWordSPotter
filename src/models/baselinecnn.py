import torch
import torch.nn as nn

class Conv1dNet(torch.nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1),  
            torch.nn.BatchNorm2d(num_features=32),  
            nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1),  
            torch.nn.BatchNorm2d(num_features=32),  
            nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        
        self.classifier = nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            torch.nn.Linear(16, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x