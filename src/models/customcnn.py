import torch

class CustomNet(torch.nn.Module):
    def __init__(self, n_classes: int, in_features: int = 1):
        super().__init__()

        self.activation = torch.nn.SELU()
        self.conv_1 = torch.nn.Conv2d(
            in_channels=in_features,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,  # Добавляем padding для сохранения размера
        )
        self.avgpool = torch.nn.AvgPool2d(kernel_size=2)

        self.conv_2 = torch.nn.Conv2d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_3 = torch.nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_4 = torch.nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_5 = torch.nn.Conv2d(
            in_channels=16,
            out_channels=11,
            kernel_size=1,
            stride=1,
        )
        # Добавляем адаптивный пулинг для приведения к нужному размеру
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((2, 2))
        self.linear = torch.nn.Linear(44, n_classes)  # 11 каналов * 2 * 2 = 44
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
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
        x = self.adaptive_pool(x)  # Приводим к размеру (2, 2)
        x = self.flatten(x)
        x = self.activation(x)
        x = self.linear(x)

        return x