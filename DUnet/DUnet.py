import torch
import torch.nn as nn
#import torchvision.transforms.functional as F

class Attn(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.filter_x = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(num_features=channels)
        )
        self.filter_g = nn.Sequential(
            nn.Conv3d(2 * channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(num_features=channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.filter_1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(num_features=channels),
            nn.Sigmoid()
        )

    def forward(self, x1, g):
        x = self.filter_x(x1)
        g = self.filter_g(g)
        g = nn.functional.interpolate(g, size=x.size()[2:], mode='trilinear', align_corners=False)
        out = x + g
        out = self.relu(out)
        out = self.filter_1(out)
        out = x1 * out

        return out


class DoubleConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sequential = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.sequential(x)


class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.double_conv1 = DoubleConvLayer(in_channels=18, out_channels=36)
        self.double_conv2 = DoubleConvLayer(in_channels=36, out_channels=72)
        self.double_conv3 = DoubleConvLayer(in_channels=72, out_channels=144)
        self.double_conv4 = DoubleConvLayer(in_channels=144, out_channels=72)
        self.double_conv5 = DoubleConvLayer(in_channels=72, out_channels=144)
        self.double_conv6 = DoubleConvLayer(in_channels=144, out_channels=72)
        self.double_conv7 = DoubleConvLayer(in_channels=72, out_channels=36)
        self.double_conv8 = DoubleConvLayer(in_channels=144, out_channels=288)
        self.double_conv9 = DoubleConvLayer(in_channels=288, out_channels=144)
        self.double_conv10 = DoubleConvLayer(in_channels=144, out_channels=288)
        self.double_conv11 = DoubleConvLayer(in_channels=288, out_channels=144)
        self.max_pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.out_activation = nn.Sigmoid()
        self.deconv1 = nn.ConvTranspose3d(in_channels=144, out_channels=72, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose3d(in_channels=144, out_channels=72, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose3d(in_channels=72, out_channels=36, kernel_size=2, stride=2)
        self.deconv4 = nn.ConvTranspose3d(in_channels=288, out_channels=144, kernel_size=2, stride=2, output_padding=1)
        self.deconv5 = nn.ConvTranspose3d(in_channels=288, out_channels=144, kernel_size=2, stride=2, output_padding=1)
        self.final_conv = nn.ConvTranspose3d(in_channels=36, out_channels=1, kernel_size=1)
        self.attn1 = Attn(36)
        self.attn2 = Attn(72)
        self.attn3 = Attn(144)
        self.attn4 = Attn(72)
        self.attn5 = Attn(64)

    def forward(self, x):
        x = self.double_conv1(x)
        identity1 = x
        x = self.max_pool(x)

        x = self.double_conv2(x)
        identity2 = x
        x = self.max_pool(x)

        x = self.double_conv3(x)
        identity3 = x
        x = self.max_pool(x)

        x = self.double_conv8(x)
        identity3 = self.attn3(identity3, x)
        x = self.deconv4(x)
        x = torch.cat((x, identity3), dim=1)
        x = self.double_conv9(x)

        identity2 = self.attn2(identity2, x)
        x = self.deconv1(x)

        x = torch.cat((x, identity2), dim=1)
        x = self.double_conv4(x)
        identity4 = x
        x = self.max_pool(x)

        x = self.double_conv5(x)
        identity5 = x
        x = self.max_pool(x)

        x = self.double_conv10(x)

        identity5 = self.attn3(identity5, x)
        x = self.deconv5(x)
        x = torch.cat((x, identity5), dim=1)
        x = self.double_conv11(x)

        identity4 = self.attn4(identity4, x)
        x = self.deconv2(x)
        x = torch.cat((x, identity4), dim=1)
        x = self.double_conv6(x)

        identity1 = self.attn1(identity1, x)
        x = self.deconv3(x)
        x = torch.cat((x, identity1), dim=1)
        x = self.double_conv7(x)

        x = self.final_conv(x)

        return self.out_activation(x)


if __name__ == "__main__":
    device = torch.device("cuda")
    model = UNet().to(device)
    x = torch.randn(1, 18, 36, 36, 36).to(device)
    y = model(x)
