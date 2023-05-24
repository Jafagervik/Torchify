#!/usr/bin/python3

import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from tqdm import tqdm
import numpy as np
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD


# LOAD SOME COOL DATA

dtype = torch.float
device = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG = True

train = torch.rand(54000, 1, 28, 28, dtype=dtype, device=device)
test = torch.rand(6000, 1, 28, 28, dtype=dtype, device=device)


class OmegaNet(Module):

    def __init__(self):
        super().__init__()

        self.b1 = self.conv_block(1, 4)
        self.b2 = self.conv_block(4, 4)
        self.l1 = Linear(4*7*7, 10)

    def conv_block(self, x_in, x_out):
        self.conv1 = Conv2d(x_in, x_out, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = BatchNorm2d(4)
        self.r1 = ReLU(inplace=True)
        self.mp1 = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = x.view(x.size(0), -1)
        x = self.l1(x)

        return x


# ====================
#   Hyperparameters
# ====================
EPOCHS = 25
BATCHES = 20
LR = 1e-6


model = OmegaNet()

optimizer = Adam(model.parameters(), lr=LR)

# Loss function to use
criterion = CrossEntropyLoss()

model.to(device)
criterion.to(device)

if (DEBUG):
    print(model)


def train(epoch: int):
    model.train()

    loss = 0.0

    xt, yt = Variable(xt), Varible(yt)

    xv, yv = Variable(xv), Variable(yv)

    xt, yt, xv, yv = xt.to(device), yt.to(device), xv.to(device), yv.to(device)

    # =====================================
    #   Clearing the gradients of the model param
    # ============================================
    optimizer.zero_grad()

    ot = model(xt)
    ov = model(xv)

    loss_train = criterion(ot, yt)
    loss_test = criterion(ov, yv)
    train_losses.append(loss_train)
    test.append(loss_test)

    loss_train.backward()
    optimizer.step()
    loss = loss_train.item()

    if epoch % 2 == 0:
        print(f"Epoch : {epoch} loss: {loss_test}")


train_losses = []
test_losses = []

for epoch in tqdm(range(EPOCHS)):
    train(epoch)


def plotting():
    plt.plot(train_losses, label="Train loss")
    plt.plot(test_losses, label="Val loss")
    plt.legend()
    plt.show()


if DEBUG:
    plotting()
