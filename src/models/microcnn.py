import torch.nn as nn

class MicroCNN(nn.Module):
    def __init__(self, num_classes=10, in_features=64):
        super(MicroCNN, self).__init__()
        
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            nn.Conv1d(in_features, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        # Заглушка для классификатора
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
                
        self.classifier = nn.Sequential(
                nn.Linear(8, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, self.num_classes)
            ) 
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
            
        x = self.classifier(x)
        return x