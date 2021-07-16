from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

def open_image(path, transform, device):
    img = Image.open(path).convert("RGB")
    return transform(img).to(device)


class CycleDataset(Dataset):
    def __init__(self, image_path, transform=None, device="cuda"):
        self.image_path = image_path
        self.transform = transform
        self.device = device
        self.images = os.listdir(image_path)
        self.length = len(self.images)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = self.images[index]
        path = os.path.join(self.image_path, img)
        img = open_image(path, self.transform, self.device)

        return img
