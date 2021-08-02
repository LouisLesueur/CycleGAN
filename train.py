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
        '--batch_size',
        type=int,
        help='batch_size',
        default=2
        )

parser.add_argument(
        '--weight_decay',
        type=float,
        help='weight_decay',
        default=0.5
        )

parser.add_argument(
        '--im1_dir',
        type=str,
        help='Type 1 image directory',
        default='data/image1'
        )

parser.add_argument(
        '--im2_dir',
        type=str,
        help='Type 2 image directory',
        default='data/image2'
        )

parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate',
        default=1e-05
        )

parser.add_argument(
        '--lambda_cycle',
        type=float,
        help='Lambda cycle (for the loss)',
        default=10
        )

parser.add_argument(
        '--num_workers',
        type=int,
        help='Num workers',
        default=4
        )

parser.add_argument(
        '--epochs',
        type=int,
        help='Number of epochs',
        default=50
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

parser.add_argument(
        '--save_model',
        type=bool,
        help='Save a model ?',
        default=False
        )

parser.add_argument(
        '--checkpoint_dir',
        type=str,
        help='checkpoint directory',
        default='checkpoints'
        )

parser.add_argument(
        '--use_optuna',
        type=bool,
        help='use optuna to find best hyperparams',
        default=False
        )

args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"


def optuna_objective(trial):

    args.lr = trial.suggest_uniform("lr",0.00001, 0.0002)
    args.weight_decay = trial.suggest_uniform("weight_decay",0, 1)
    best = 100000
    model = CycleGAN(device, args.checkpoint_dir)

    if args.load_model is not None:
        model.load_checkpoint(args.load_model)

    for epoch in range(args.epochs):
        intermediate_value = model.train(args)
        if intermediate_value < best:
            best = intermediate_value
        trial.report(intermediate_value, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return best



if __name__ == '__main__':

    if args.use_optuna:
        args.epochs = 5
        sampler = optuna.samplers.TPESampler(multivariate=True)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2, interval_steps=1)
        study = optuna.create_study(storage='sqlite:///base.db',
                                    pruner=pruner, sampler=sampler,
                                    study_name="cycleGAN_parameter_study")
        study.optimize(optuna_objective, n_trials=100)

    else:
        model = CycleGAN(device, args.checkpoint_dir)

        if args.load_best:
            model.load_best()

        if args.load_model is not None:
            model.load_checkpoint(args.load_model)

        for epoch in range(args.epochs):
            model.train(args)
            if args.save_model:
                model.save_checkpoint()
