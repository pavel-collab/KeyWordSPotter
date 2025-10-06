import torch

class CustomNet(torch.nn.Module):
    def __init__(self, num_classes: int, in_features: int = 64):
        super().__init__()

        self.activation = torch.nn.SELU()
        self.conv_1 = torch.nn.Conv1d(
            in_channels=in_features,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.avgpool = torch.nn.AvgPool1d(kernel_size=2)

        self.conv_2 = torch.nn.Conv1d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_3 = torch.nn.Conv1d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_4 = torch.nn.Conv1d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_5 = torch.nn.Conv1d(
            in_channels=16,
            out_channels=11,
            kernel_size=1,
            stride=1,
        )
        self.adaptive_pool = torch.nn.AdaptiveAvgPool1d(2)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(22, num_classes)  # 11 каналов * 2 = 22

    def forward(self, x):
        # # x shape: (batch, n_mels, n_times)
        # # Если нужно добавить dimension для каналов (если in_channels=1)
        # if x.dim() == 2:
        #     x = x.unsqueeze(1)  # (batch, 1, n_times)
        
        x = self.conv_1(x)
        x = self.activation(x)
        x = self.avgpool(x)

        x = self.conv_2(x)
        x = self.activation(x)
        x = self.avgpool(x)

        x = self.conv_3(x)
        x = self.activation(x)
        x = self.avgpool(x)

        x = self.conv_4(x)
        x = self.activation(x)
        x = self.avgpool(x)

        x = self.conv_5(x)
        x = self.adaptive_pool(x)  # Приводим к размеру (batch, 11, 2)
        x = self.flatten(x)        # (batch, 22)
        x = self.activation(x)
        x = self.linear(x)

        return x