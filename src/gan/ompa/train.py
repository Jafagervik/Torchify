#!/usr/bin/python


import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import utils
from model import OmegaNet
from math import log2
from tqdm import tqdm
import config

torch.backends.cudnn.benchmarks = True


def get_data_loader(img_size):
    """
    Gets us both the dataloader and the dataaset

    img_size
    """
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5 for _ in range(config.CHANNELS_IMG)],
                [0.5 for _ in range(config.CHANNELS_IMG)],
            ),
        ]
    )

    batch_size = config.BATCH_SIZES[int(log2(img_size / 4))]
    dataset = datasets.ImageFolder(root=config.DATASET, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    return loader, dataset


def train_step():
    pass


def test_step():
    pass


def main():
    model = OmegaNet().to(config.DEVICE)

    optimizer = optim.Adam(
        model.parameters, lr=config.LEARNING_RATE, betas=(0.0, 0.99))

    scalar = torch.cuda.amp.GradScaler()

    writer = SummaryWriter(f"logs/{model.__name__}")

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, model, optimizer,
                        config.LEARNING_RATE)  # from utils

    model.train()

    tb_step = 0

    step = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))
    for epoch in range(config.EPOCHS):
        alpha = 1e-5

        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer, filename=config.CHECKPOINT_GEN)

        step += 1


if __name__ == "__main__":
    pass
