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
from nets import Generator, Discriminator
from tqdm import tqdm
from dataset import CycleDataset 
import os


class CycleGAN:

    def __init__(self, device, checkpoint_path="checkpoints"):
        
        self.device = device

        self.checkpoint_path = checkpoint_path

        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
            os.mkdir(os.path.join(checkpoint_path, 'best'))

        if not os.path.exists(os.path.join(checkpoint_path, 'best')):
            os.mkdir(os.path.join(checkpoint_path, 'best'))

        self.mean = torch.tensor([0.5, 0.5, 0.5], device=self.device)
        self.std = torch.tensor([0.5, 0.5, 0.5], device=self.device)

        self.writer = SummaryWriter()
        self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std)
                ]
            )

        self.D1 = Discriminator(in_channels=3).to(self.device)
        self.D2 = Discriminator(in_channels=3).to(self.device)

        self.G1 = Generator(img_channels=3, num_residuals=9).to(self.device)
        self.G2 = Generator(img_channels=3, num_residuals=9).to(self.device)
        
        self.epoch = 0
        self.iteration = 0

        self.best_epoch = 0
        self.best_loss = 10000



    def save_checkpoint(self, is_best=False):
        if is_best:
            path = os.path.join(self.checkpoint_path, 'best')
        else:
            print("[CycleGAN] Saving checkpoint")
            path = os.path.join(self.checkpoint_path, f'epoch_{self.epoch}')
        
        if os.path.exists(path):
            for file in os.listdir(path):
                os.remove(os.path.join(path,file))
        else:
            os.mkdir(path)
     
        for model in ["D1", "D2", "G1", "G2"]:
            checkpoint_name = f"{model}_{self.epoch}_{self.iteration}.pth.tar"
            if is_best:
                checkpoint_name = "best_"+checkpoint_name
            checkpoint = {"state_dict": eval(f"self.{model}.state_dict()")}
            torch.save(checkpoint, os.path.join(path, checkpoint_name))


    def load_checkpoint(self, checkpoint_path):
        print("[CycleGAN] Loading checkpoint")

        for file in os.listdir(checkpoint_path):
            model_str, epoch_str, iter_str = file.removeprefix("best_").removesuffix(".pth.tar").split("_")

            checkpoint = eval(f"torch.load(os.path.join(checkpoint_path, file), map_location=self.device)")
            eval(f"self.{model_str}.load_state_dict(checkpoint[\"state_dict\"])")

            self.epoch = int(epoch_str)
            self.iteration = int(iter_str)

        self.best_epoch = self.epoch
        self.best_loss = 100000


    def load_best(self):
        self.load_checkpoint(os.path.join(self.checkpoint_path, 'best'))


    def convert(self, im_path, out_path, im1_t_im2=True):

        if im1_t_im2==True:
            function = "self.G2"
        else:
            function = "self.G1"

        data = CycleDataset(image_path = im_path,
                            transform = self.transform,
                            device = self.device)


        loader = DataLoader(data, batch_size = 1, shuffle = False)
        loop = tqdm(loader, leave=True)
        print(f"[CycleGAN] converting {len(loader)} images from {im_path} to {out_path}...")
        
        eval(f"{function}.eval()")
        
        for idx, im in enumerate(loop):
            output = eval(f"{function}(im)")
            save_image(0.5*output+0.5, os.path.join(out_path, f"im_{idx}.png"))


    def train(self, args):
        
        for function in ["self.D1", "self.D2", "self.G1", "self.G2"]:
            eval(f"{function}.train()")

        train_cycle_loss = 0
        train_style_loss = 0

        optD = optim.Adam(
                list(self.D1.parameters()) + list(self.D2.parameters()),
                weight_decay=args.weight_decay,
                lr = args.lr,
                betas = (0.5, 0.999)
                )

        optG = optim.Adam(
                list(self.G1.parameters()) + list(self.G2.parameters()),
                weight_decay=args.weight_decay,
                lr = args.lr,
                betas = (0.5, 0.999)
                )

        l1 = nn.L1Loss()
        mse = nn.MSELoss()

        data1 = CycleDataset(image_path = args.im1_dir,
                            transform = self.transform,
                            device = self.device)

        data2 = CycleDataset(image_path = args.im2_dir,
                             transform = self.transform,
                             device = self.device)

        loader1 = DataLoader(data1, batch_size = 1, shuffle = False)
        loader2 = DataLoader(data2, batch_size = 1, shuffle = False)

        scalerG = torch.cuda.amp.GradScaler()
        scalerD = torch.cuda.amp.GradScaler()
 
        loop = tqdm(loader1, leave=True)

        for idx1, im1 in enumerate(loop):
            for idx2, im2 in enumerate(loader2):

                with torch.cuda.amp.autocast():
        
                    fake1 = self.G1(im2)
                    D1_real = self.D1(im1)
                    D1_fake = self.D1(fake1.detach())
                    
                    D1_real_loss = mse(D1_real, torch.ones_like(D1_real))
                    D1_fake_loss = mse(D1_fake, torch.zeros_like(D1_fake))
                    D1_loss = D1_real_loss + D1_fake_loss

                    fake2 = self.G2(im1)
                    D2_real = self.D2(im2)
                    D2_fake = self.D2(fake2.detach())
                    
                    D2_real_loss = mse(D2_real, torch.ones_like(D2_real))
                    D2_fake_loss = mse(D2_fake, torch.zeros_like(D2_fake))
                    D2_loss = D2_real_loss + D2_fake_loss

                    style_loss = (D1_loss + D2_loss)/2

                optD.zero_grad()
                scalerD.scale(style_loss).backward()
                scalerD.step(optD)
                scalerD.update()

                with torch.cuda.amp.autocast():

                    D1_fake = self.D1(fake1)
                    D2_fake = self.D2(fake2)

                    loss_G1 = mse(D1_fake, torch.ones_like(D1_fake))
                    loss_G2 = mse(D2_fake, torch.ones_like(D2_fake))

                    cycle1 = self.G1(fake2)
                    cycle2 = self.G2(fake1)
                    cycle1_loss = l1(im1, cycle1)
                    cycle2_loss = l1(im2, cycle2)

                    cycle_loss = (loss_G1
                                + loss_G2
                                + cycle1_loss * args.lambda_cycle
                                + cycle2_loss * args.lambda_cycle)

                optG.zero_grad()
                scalerG.scale(cycle_loss).backward()
                scalerG.step(optG)
                scalerG.update()

                if cycle_loss.item() < self.best_loss:
                    self.best_loss = cycle_loss.item()
                    self.best_epoch = self.epoch
                    self.save_checkpoint(is_best=True)

                train_cycle_loss += cycle_loss.item()
                train_style_loss += style_loss.item()

                self.writer.add_scalar("Loss/Style", style_loss, self.iteration)
                self.writer.add_scalar("Loss/Cycle", cycle_loss, self.iteration)

                if idx1 % 200  == 0:
                    self.writer.add_image(f"im2_gen/{idx1}", fake2[0]*0.5+0.5, global_step = self.iteration)
                    self.writer.add_image(f"im1_cycle/{idx1}", cycle1[0]*0.5+0.5, global_step=self.iteration)
                if idx2 % 200 == 0:
                    self.writer.add_image(f"im1_gen/{idx2}", fake1[0]*0.5+0.5, global_step=self.iteration)
                    self.writer.add_image(f"im2_cycle/{idx2}", cycle2[0]*0.5+0.5, global_step=self.iteration)

                self.iteration += 1

        self.epoch += 1
        print(f"[CycleGAN] Training: epoch {self.epoch} finished, train_style_loss: {train_style_loss/(len(data1)*len(data2))}, train cycle_loss: {train_cycle_loss/(len(data1)*len(data2))}")
        self.writer.add_scalar("TotalLoss/Cycle", train_cycle_loss/(len(data1)*len(data2)), self.epoch)
        self.writer.add_scalar("TotalLoss/Style", train_style_loss/(len(data1)*len(data2)), self.epoch)
        return train_style_loss/(len(data1)*len(data2))
