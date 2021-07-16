import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
import argparse
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from cycleGAN import CycleGAN
from tqdm import tqdm
from dataset import CycleDataset
import os
import optuna

parser = argparse.ArgumentParser(description='Simple Neural Style Transfer Implementation')

parser.add_argument(
        '--input_dir',
        type=str,
        help='Image to convert/to be converted of type 1',
        )

parser.add_argument(
        '--output_dir',
        type=str,
        help='Image to convert/to be converted of type 2',
        )

parser.add_argument(
        '--im1_t_im2',
        type=bool,
        help="wanted conversion",
        default=False
        )

parser.add_argument(
        '--load_best',
        type=bool,
        help='Load last best model ?',
        default=False
        )

parser.add_argument(
        '--load_model',
        type=str,
        help='path to a model to load',
        default=None
        )


args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"



if __name__ == '__main__':

    model = CycleGAN(device)

    if args.load_best:
        model.load_best()

    if args.load_model is not None:
        model.load_checkpoint(args.load_model)

    print("Im1_t_im2 ", args.im1_t_im2)
    model.convert(args.input_dir, args.output_dir, args.im1_t_im2)
