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
from generator import Generator
from discriminator import Discriminator
from tqdm import tqdm
from dataset import CycleDataset

writer = SummaryWriter()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_GEN_1 = "gen1.pth.tar"
CHECKPOINT_GEN_2 = "gen2.pth.tar"
CHECKPOINT_CRITIC_1 = "critic1.pth.tar"
CHECKPOINT_CRITIC_2 = "critic2.pth.tar"


parser = argparse.ArgumentParser(description='Simple Neural Style Transfer Implementation')


parser.add_argument(
        '--batch_size', 
        type=int, 
        help='image size',
        default=1
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
        default=0.0002
        )

parser.add_argument(
        '--lambda_identity', 
        type=float, 
        help='Lambda Identity',
        default=0.0
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
        '--load_model', 
        type=bool, 
        help='Load a model ?',
        default=False
        )

parser.add_argument(
        '--save_model', 
        type=bool, 
        help='Save a model ?',
        default=True
        )

args = parser.parse_args()

MEAN = torch.tensor([0.5, 0.5, 0.5], device=DEVICE)
STD = torch.tensor([0.5, 0.5, 0.5], device=DEVICE)

TRANSFORM = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ]
    )


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr



def train(D1, D2, G1, G2, loader, optD, optG, l1, mse, scalerD, scalerG):

    real1 = 0
    fakes1 = 0
    loop = tqdm(loader, leave=True)

    for idx, (im1, im2) in enumerate(loop):

        with torch.cuda.amp.autocast():
            
            fake1 = G1(im1)
            D1_real = D1(im1)
            D1_fake = D1(fake1.detach())
            real1 += D1_real.mean().item()
            fakes1 += D1_fake.mean().item()
            D1_real_loss = mse(D1_real, torch.ones_like(D1_real))
            D1_fake_loss = mse(D1_fake, torch.zeros_like(D1_fake))
            D1_loss = D1_real_loss + D1_fake_loss

            fake2 = G2(im2)
            D2_real = D2(im2)
            D2_fake = D2(fake2.detach())
            D2_real_loss = mse(D2_real, torch.ones_like(D2_real))
            D2_fake_loss = mse(D2_fake, torch.zeros_like(D2_fake))
            D2_loss = D2_real_loss + D2_fake_loss

            D_loss = (D1_loss + D2_loss)/2

        optD.zero_grad()
        scalerD.scale(D_loss).backward()
        scalerD.step(optD)
        scalerD.update()

        with torch.cuda.amp.autocast():

            D1_fake = D1(fake1)
            D2_fake = D2(fake2)

            loss_G1 = mse(D1_fake, torch.ones_like(D1_fake))
            loss_G2 = mse(D2_fake, torch.ones_like(D2_fake))

            cycle1 = G1(im2)
            cycle2 = G2(im1)
            cycle1_loss = l1(im1, cycle1)
            cycle2_loss = l1(im2, cycle2)

            id1 = G1(im1)
            id2 = G2(im2)
            id1_loss = l1(im1, cycle1)
            id2_loss = l1(im2, cycle2)

            G_loss = (
                    loss_G1
                    + loss_G2
                    + cycle1_loss * args.lambda_cycle
                    + cycle2_loss * args.lambda_cycle
                    + id1_loss * args.lambda_identity
                    + id2_loss * args.lambda_identity
                    )
        optG.zero_grad()
        scalerG.scale(G_loss).backward()
        scalerG.step(optG)
        scalerG.update()

        writer.add_scalar("Loss/Discriminator", D_loss, idx)
        writer.add_scalar("Loss/Generator", G_loss, idx)

        if idx % 200 == 0:
            save_image(fake1*0.5+0.5, f"saved_images/im1_{idx}.png")
            save_image(fake2*0.5+0.5, f"saved_images/im2_{idx}.png")

        loop.set_postfix(r1=real1/(idx+1), f1=fakes1/(idx+1))

            

if __name__ == '__main__':

    D1 = Discriminator(in_channels=3).to(DEVICE)
    D2 = Discriminator(in_channels=3).to(DEVICE)

    G1 = Generator(img_channels=3, num_residuals=9).to(DEVICE)
    G2 = Generator(img_channels=3, num_residuals=9).to(DEVICE)

    optD = optim.Adam(
            list(D1.parameters()) + list(D2.parameters()),
            lr = args.lr,
            betas = (0.5, 0.999)
            )
    
    optG = optim.Adam(
            list(G1.parameters()) + list(G2.parameters()),
            lr = args.lr,
            betas = (0.5, 0.999)
            )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if args.load_model:
        load_checkpoint(CHECKPOINT_GEN_1, G1, optG, args.lr)
        load_checkpoint(CHECKPOINT_GEN_2, G2, optG, args.lr)
        load_checkpoint(CHECKPOINT_CRITIC_1, D1, optD, args.lr)
        load_checkpoint(CHECKPOINT_CRITIC_2, D2, optD, args.lr)


    data = CycleDataset(image1_path = args.im1_dir,
                        image2_path = args.im2_dir,
                        transform = TRANSFORM,
                        device = DEVICE)


    loader = DataLoader(
            data,
            batch_size = 1,
            shuffle = False,
            )

    scalerG = torch.cuda.amp.GradScaler()
    scalerD = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        train(D1, D2, G1, G2, loader, optD, optG, L1, mse, scalerD, scalerG)

        if args.save_model:
            save_checkpoint(G1, optG, filename=CHECKPOINT_GEN_1)
            save_checkpoint(G2, optG, filename=CHECKPOINT_GEN_2)
            save_checkpoint(D1, optD, filename=CHECKPOINT_CRITIC_1)
            save_checkpoint(D2, optD, filename=CHECKPOINT_CRITIC_2)
