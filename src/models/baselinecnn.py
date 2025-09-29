import torch
import torch.nn as nn

class Conv1dNet(torch.nn.Module):
    def __init__(self, num_classes=5, in_features=64):
        super().__init__()
        self.features = nn.Sequential(
            torch.nn.Conv1d(in_channels=in_features, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1),  
            torch.nn.BatchNorm1d(num_features=32),  
            nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2),
            
            torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1),  
            torch.nn.BatchNorm1d(num_features=32),  
            nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2),
        )
        
        self.classifier = nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            torch.nn.Linear(16, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x