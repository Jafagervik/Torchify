import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import math

from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
dtype = torch.float

"""
1. get data
2. normalize
3. set up hyperparams, aka adjustable params
4. train step, this should do backprop gradient descent
5. test should not backprop
6. plot data, train and test, acc and loss - maybe even auc and mAP
7. profit


what is needed

model, loss function, optimizer, actual data, grad descent

with torch.autograd?
means that we dont need to implement
multiple layers of backpropagation by hand for larger models
"""


class MembraneNet(Module):

    def __init__(self, numChannels, classes):
        super().__init__()
        self.c1 = Conv2d(in_channels=numChannels,
                         out_channels=20, kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.mp1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.c2 = Conv2d(in_channels=20,
                         out_channels=50, kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.mp2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=800, out_features=500)
        self.relu3 = ReLU()
        # initialize our softmax classifier
        self.fc2 = Linear(in_features=500, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, X):
        X = self.c1(X)
        X = self.relu1(X)
        X = self.mp1(X)

        X = self.c2(X)
        X = self.relu2(X)
        X = self.mp2(X)

        X = flatten(X, 1)
        X = self.fc1(X)
        X = self.relu3(X)

        X = self.fc2(X)
        return self.logSoftmax(X)


model = MembraneNet().to(device)
print(model)


X = torch.rand(1, 28, 28)
ls = model(X)


def main():
    pass


if __name__ == "__main__":
    main()
