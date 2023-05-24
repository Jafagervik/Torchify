#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import os
# import sys
from tqdm import tqdm


class Discriminator(nn.Module):

    def __init__(self, in_features):
        super().__init__()

        self.l1 = nn.Linear(in_features, 128)
        self.l2 = nn.Linear(128, 1)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.sigma = nn.Sigmoid()

    def forward(self, X):
        X = self.l1(X)
        X = self.leaky_relu(X)
        X = self.l2(X)
        X = self.sigma(X)

        return X


class Generator(nn.Module):
    def __init__(self, zdim, imgdim):
        super().__init__()
        self.l1 = nn.Linear(zdim, 256)
        self.l2 = nn.Linear(256, imgdim)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.tanh = nn.Tanh()

    def forward(self, X):
        X = self.l1(X)
        X = self.leaky_relu(X)
        X = self.l2(X)
        X = self.tanh(X)

        return X


device = "cuda" if torch.cuda.is_available() else "cpu"


lr = 3e-4  # Quote from Andrej Karpathy: Best lr for ADAM optimizer
zdim = 64
imgdim = 28 * 28 * 1
dropout = 0.02

batch_size = 32
EPOCHS = 50


disc = Discriminator(imgdim).to(device)
gen = Generator(zdim, imgdim).to(device)


fixed_noise = torch.rand((batch_size, zdim)).to(device)

transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

optd = optim.Adam(disc.parameters(), lr=lr)
optg = optim.Adam(gen.parameters(), lr=lr)

# Loss function

criterion = nn.BCELoss()

writer_fake = SummaryWriter(f"runs/GANMNIST/fake")
writer_real = SummaryWriter(f"runs/GANMNIST/real")

step = 0

for epoch in tqdm(range(EPOCHS)):
    for batchidx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batchsize = real.shape[0]

        # Train disc max log(D(readl)) + log(1-D(G(z)))
        noise = torch.randn(batchsize, zdim).to(device)
        fake = gen(noise)

        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        lossD = (lossD_real + lossD_fake) / 2

        disc.zero_grad()
        lossD.backward(retain_graph=True)
        optd.step()

        # Train Generator min log(1 - D(G(z)))  <==> max log(D(G(z)))

        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        optg.step()

        if batchidx == 0:
            print(
                f"Epoch [{epoch}/{EPOCHS}] Loss D: \
                        {lossD:.4f}, Loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(
                    fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(
                    data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )

                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )

                step += 1


# NOTE: Do this to avoid any buffersss
writer_real.flush()
writer_fake.flush()
