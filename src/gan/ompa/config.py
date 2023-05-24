"""
Hyperparameters to use for a setup or project
"""

import torch


NUM_WORKERS = 4
# Which device to run on
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Number of epochs you want to use
EPOCHS = 50
# Save model after training
SAVE_MODEL = True
# Load model from checkpoint
LOAD_MODEL = False
# At what rate should the optimizer learn
LEARNING_RATE = 3e-4
# Relative path to dataset you're currently working with
DATASET = ""
# Number of channels going in to the network
IN_CHANNELS = 512
# Number of z dimensions
Z_DIM = 22
# Batch sizes
BATCH_SIZES = [32, 32, 32, 32, 32, 32, 16, 16, 16, 8, 4]
# TODO: dont know
CRITIC_ITERATIONS = 1
# Noise to use at start
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
# How many color channels an image has. 3 for RGB, 1 for BW
CHANNELS_IMG = 3
# When should we start training
START_TRAIN_AT_IMG_SIZE = 128

CHECKPOINT_GEN = ""
CHECKPOINT_CRITIC = ""
# Number of
LAMBDA_GP = 10
# Scalable epochs
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES)
