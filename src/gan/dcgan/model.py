#!/usr/bin/python
"""
Hello File!
"""

import torch
from torch import nn
from config import CHANNELS_IMG


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int, padding: int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=False)
        self.activation = nn.LeakyReLU(0.2)
        # self.mp = nn.MaxPool2d(kernel_size=2)
        # self.bn = nn.BatchNormalize()

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        x = self.activation(x)
        # x = self.mp(x)

        return x


class ConvBlockT(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int, padding: int):
        super().__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                                       kernel_size, stride, padding, bias=False)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super().__init__()

        self.net = nn.Sequential(
            # Start
            ConvBlock(channels_img, features_d, 4, 2, 1),

            ConvBlock(features_d, features_d * 2, 4, 2, 1),
            ConvBlock(features_d * 2, features_d * 4, 4, 2, 1),
            ConvBlock(features_d * 4, features_d * 8, 4, 2, 1),

            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super().__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            ConvBlockT(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            ConvBlockT(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            ConvBlockT(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            ConvBlockT(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, CHANNELS_IMG, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success, tests passed!")


if __name__ == "__main__":
    test()
