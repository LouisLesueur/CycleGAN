from PIL import Image
import os
from torch.utils.data import Dataset

class CycleDataset(Dataset):
    def __init__(self, image1_path, image2_path, transform=None, device="cuda"):
        self.image1_path = image1_path
        self.image2_path = image2_path
        self.transform = transform
        self.device = device

        self.images1 = os.listdir(image1_path)
        self.images2 = os.listdir(image2_path)

        self.length = max(len(self.images1), len(self.images2))
        self.len1 = len(self.images1)
        self.len2 = len(self.images2)


    def __len__(self):
        return self.length

    def __getitem__(self, index):

        img1 = self.images1[index % self.len1]
        img2 = self.images2[index % self.len2]

        path1 = os.path.join(self.image1_path, img1)
        path2 = os.path.join(self.image2_path, img2)

        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")

        img1 = self.transform(img1).to(self.device)
        img2 = self.transform(img2).to(self.device)

        return img1,img2
