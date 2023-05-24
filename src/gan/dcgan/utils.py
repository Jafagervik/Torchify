import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
import torch

root_dir = ""
files = os.listdir(root_dir)


def resize_image(f, size, folder_to_save):
    image = Image.open(root_dir + f).resize((size, size), Image.LANCZOS)
    image.save(folder_to_save+f, quality=100)


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old
    # checkpoint and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def plot_gen_and_orig_image(out_g, real_img):
    """ Plor and show real vs generated image

    out_g: Image the generator has produced
    real_img: Image from the original dataset
    """
    plt.figure(figsize=(8, 8))

    plt.subplot(1, 2, 1)
    plt.title("Generated Image")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(np.transpose(out_g.resize(
        3, 32, 32).cpu().detach().numpy(), (1, 2, 0)))

    plt.subplot(1, 2, 2)
    plt.title("Original Image")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(np.transpose(real_img.numpy(), (1, 2, 0)))
    plt.show()
