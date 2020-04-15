import torch
import torch.nn as nn
import torch.nn.functional as F


class VT_CNN2(torch.nn.Module):

    def __init__(
        self,
        n_classes: int = 10,
        dropout: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super(VT_CNN2, self).__init__()

        self.device = device
        self.loss = nn.CrossEntropyLoss()

        self.model = nn.Sequential(
            nn.ZeroPad2d(padding=(2, 2, 0, 0,)),    # zero pad front/back of each signal by 2
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(1, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.ZeroPad2d(padding=(2, 2, 0, 0,)),    # zero pad front/back of each signal by 2
            nn.Conv2d(in_channels=256, out_channels=80, kernel_size=(2, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Flatten(),
            nn.Linear(in_features=10560, out_features=256, bias=True),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=256, out_features=n_classes, bias=True),
        )

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        x = x.to(self.device)
        y_pred = self.model(x)
        y_pred = y_pred.to("cpu")
        y_pred = torch.softmax(y_pred, dim=-1)
        values, indices = torch.max(y_pred, dim=-1)
        indices = indices.numpy()
        return indices


class ResidualBlock(nn.Module):
    """Base Residual Block
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        activation
    ):

        super(ResidualBlock, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            activation(),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x) + x
        return x


class MRResNet(torch.nn.Module):
    """Modulation Recognition ResNet (Mine)"""
    def __init__(
        self,
        n_classes: int = 10,
        n_res_blocks: int = 16,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super(MRResNet, self).__init__()

        self.n_classes = n_classes
        self.n_res_blocks = n_res_blocks
        self.device = device
        self.loss = nn.CrossEntropyLoss()

        self.head = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.ReLU()
        )

        # Residual Blocks
        self.res_blocks = [
            ResidualBlock(channels=64, kernel_size=3, activation=nn.ReLU) \
            for _ in range(self.n_res_blocks)
            ]
        self.res_blocks.append(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        self.res_blocks = nn.Sequential(*self.res_blocks)

        # Output layer
        self.tail = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=64, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=n_classes, bias=True),
        )

    def forward(self, x):
        x = x.squeeze()
        x = self.head(x)
        shortcut = x
        x = self.res_blocks(x) + shortcut

        # Global average pooling
        x = torch.mean(x, dim=-1)

        # Classification
        x = self.tail(x)
        return x

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        x = x.to(self.device)
        y_pred = self.forward(x)
        y_pred = y_pred.to("cpu")
        y_pred = torch.softmax(y_pred, dim=-1)
        values, indices = torch.max(y_pred, dim=-1)
        indices = indices.numpy()
        return indices