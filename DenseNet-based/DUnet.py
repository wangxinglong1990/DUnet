import torch
import torch.nn as nn

from typing import Tuple


class DenseBlock(nn.Module):

    def __init__(
            self, in_channels: int, hidden_channels: int, out_channels: int, count: int,
    ):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        # First iteration takes different number of input channels and does not repeat
        self.layer1 = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.Conv3d(in_channels, hidden_channels, kernel_size=(1, 1, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(hidden_channels),
            nn.Conv3d(
                hidden_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)
            ),
            nn.ReLU()
        )

        # Remaining repeats are identical blocks
        self.layer2 = nn.Sequential(
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, hidden_channels, kernel_size=(1, 1, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(hidden_channels),
            nn.Conv3d(
                hidden_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)
            ),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.BatchNorm3d(out_channels * 2),
            nn.Conv3d(out_channels * 2, hidden_channels, kernel_size=(1, 1, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(hidden_channels),
            nn.Conv3d(
                hidden_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)
            ),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.BatchNorm3d(out_channels * 3),
            nn.Conv3d(out_channels * 3, hidden_channels, kernel_size=(1, 1, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(hidden_channels),
            nn.Conv3d(
                hidden_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)
            ),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.BatchNorm3d(out_channels * 4),
            nn.Conv3d(out_channels * 4, hidden_channels, kernel_size=(1, 1, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(hidden_channels),
            nn.Conv3d(
                hidden_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)
            ),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        dense2 = self.relu(torch.cat([out1, out2], 1))

        out3 = self.layer3(dense2)
        dense3 = self.relu(torch.cat([out1, out2, out3], 1))

        out4 = self.layer4(dense3)
        dense4 = self.relu(torch.cat([out1, out2, out3, out4], 1))

        out5 = self.layer5(dense4)
        dense5 = self.relu(torch.cat([out1, out2, out3, out4, out5], 1))

        return dense5


class TransitionBlock(nn.Module):

    def __init__(self, channels: int, stride=(2, 2, 2)):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            nn.Conv3d(channels, channels, kernel_size=(1, 1, 1), stride=stride),
            nn.BatchNorm3d(channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


class UpsamplingBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, size: Tuple):

        super().__init__()
        if size[0] == 2:
            d_kernel_size = 3
            d_padding = 1
        else:
            d_kernel_size = 1
            d_padding = 0

        self.upsample = nn.Upsample(
            scale_factor=size, mode="trilinear", align_corners=True
        )
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(d_kernel_size, 3, 3),
                padding=(d_padding, 1, 1),
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, projected_residual):

        residual = torch.cat(
            (self.upsample(x), self.upsample(projected_residual)),
            dim=1,
        )
        return self.conv(residual)


class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial Layers
        self.conv1 = nn.Conv3d(
            18, 96, kernel_size=(7, 7, 7), stride=1, padding=(3, 3, 3)
        )
        self.bn1 = nn.BatchNorm3d(96)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # Dense Layers
        self.dense1 = DenseBlock(96, 128, 32, 4)
        self.dense2 = DenseBlock(160, 128, 32, 4)
        self.dense3 = DenseBlock(160, 128, 32, 4)
        self.dense4 = DenseBlock(160, 128, 32, 4)
        self.tran1 = nn.Sequential(TransitionBlock(96, (1, 1, 1)), TransitionBlock(96, (2, 2, 2)))
        self.tran2 = nn.Sequential(TransitionBlock(160, (1, 1, 1)), TransitionBlock(160, (2, 2, 2)))
        self.tran3 = nn.Sequential(TransitionBlock(160, (1, 1, 1)), TransitionBlock(160, (3, 3, 3)))
        self.tran4 = nn.Sequential(
            nn.Conv3d(160, 320, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(320),
            nn.ReLU(),
            nn.Conv3d(320, 640, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(640),
            nn.ReLU(),
        )
        self.tran5 = nn.Sequential(TransitionBlock(640, (1, 1, 1)), TransitionBlock(640, (1, 1, 1)))
        self.tran6 = nn.Sequential(TransitionBlock(320, (1, 1, 1)), TransitionBlock(320, (1, 1, 1)))
        self.tran7 = nn.Sequential(TransitionBlock(160, (1, 1, 1)), TransitionBlock(160, (1, 1, 1)))
        self.tran8 = nn.Sequential(TransitionBlock(96, (1, 1, 1)), TransitionBlock(96, (1, 1, 1)))
        # Upsampling Layers
        self.upsample1 = UpsamplingBlock(640 + 640, 640, size=(1, 1, 1))
        self.upsample2 = UpsamplingBlock(640 + 160, 320, size=(3, 3, 3))
        self.upsample3 = UpsamplingBlock(320 + 160, 160, size=(2, 2, 2))
        self.upsample4 = UpsamplingBlock(160 + 160, 96, size=(2, 2, 2))
        # Final output layer
        self.conv_classifier = nn.Conv3d(96, 1, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        down1 = self.relu(self.bn1(self.conv1(x)))
        # print(down1.shape)
        down2 = self.dense1(self.tran1(down1))
        # print(down2.shape)
        down3 = self.dense2(self.tran2(down2))
        # print(down3.shape)
        down4 = self.dense3(self.tran3(down3))
        # print(down4.shape)
        base = self.tran4(down4)
        # print('base ', base.shape)

        up1 = self.upsample1(base, base)
        up1 = self.tran5(up1)
        # print(up1.shape)
        output = self.upsample2(up1, down4)
        output = self.tran6(output)
        # print(output.shape)
        output = self.upsample3(output, down3)
        output = self.tran7(output)
        # print(output.shape)
        output = self.upsample4(output, down2)
        output = self.tran8(output)
        # print(output.shape)
        output = self.sigmoid(self.conv_classifier(output))
        # print(output.shape)
        return output


if __name__ == "__main__":
    device = torch.device("cuda")
    model = DenseNet().to(device)
    x = torch.randn(1, 18, 36, 36, 36).to(device)
    y = model(x)

