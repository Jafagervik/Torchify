"""
Hello File!
"""

import torch
from torch import nn
from nn import functional as F
from math import log2


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.c1 = None
        self.activation = nn.LeakyReLU(0.1)
        self.mp = nn.MaxPool2d()
        self.bn = nn.BatchNormalize()

    def forward(self, x):
        x = self.c1(x)
        x = self.activation(x)
        x = self.mp(x)
        x = self.bn(x)

        return x


class OmegaNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


if __name__ == "__main__":
    pass
