#!/usr/bin/python


import torch
from torch import nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import utils
from model import Generator, Discriminator, initialize_weights
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
    dataloader, dataset = get_data_loader(28)  # 28 due to mnist

    disc = Discriminator(config.NOISE_DIM, config.CHANNELS_IMG,
                         config.FEATURES_GEN).to(config.DEVICE)
    gen = Generator(config.CHANNELS_IMG,
                    config.FEATURES_DISC).to(config.DEVICE)

    initialize_weights(disc)
    initialize_weights(gen)

    optimizer_disc = optim.Adam(
        disc.parameters, lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    optimizer_gen = optim.Adam(
        gen.parameters, lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    scalar = torch.cuda.amp.GradScaler()

    criterion = nn.BCELoss()

    writer_real = SummaryWriter("logs/real")
    writer_fake = SummaryWriter("logs/fake")

    if config.LOAD_MODEL:
        utils.load_checkpoint(config.CHECKPOINT_DISC, disc, optimizer_disc,
                              config.LEARNING_RATE)  # from utils
        utils.load_checkpoint(config.CHECKPOINT_GEN, gen, optimizer_gen,
                              config.LEARNING_RATE)  # from utils

    disc.train()
    gen.train()

    step = 0
    tb_step = 0

    # step = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))
    for epoch in tqdm(range(config.EPOCHS)):

        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.to(config.DEVICE)
            noise = config.FIXED_NOISE
            fake = gen(noise)

            # train disc
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

            loss_disc = (loss_disc_real + loss_disc_fake) / 2

            disc.zero_grad()
            loss_disc.backward()
            optimizer_disc.step()

            # train gen: min log(1 - D(G(z))) <--> max
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))

            gen.zero_grad()
            loss_gen.backward()
            optimizer_gen.step()

            # Print losses occasionally and print to tensorboard
            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{config.EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                      Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                )

                with torch.no_grad():
                    fake = gen(config.FIXED_NOISE)
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:32], normalize=True)

                    writer_real.add_image(
                        "Real", img_grid_real, global_step=step)
                    writer_fake.add_image(
                        "Fake", img_grid_fake, global_step=step)

                step += 1

        if config.SAVE_MODEL:
            utils.save_checkpoint(disc, optimizer_disc,
                                  filename=config.CHECKPOINT_DISC)
            utils.save_checkpoint(gen, optimizer_gen,
                                  filename=config.CHECKPOINT_GEN)


if __name__ == "__main__":
    pass
