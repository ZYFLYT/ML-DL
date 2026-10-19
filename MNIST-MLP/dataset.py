import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MNISTDataset(Dataset):
    def __init__(self, csv_file, is_train=True):
        self.data = pd.read_csv(csv_file)
        self.is_train = is_train
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        if self.is_train:
            self.labels = self.data["label"].values
            self.images = self.data.drop("label", axis=1).values.reshape(-1, 28, 28)
        else:
            self.images = self.data.values.reshape(-1, 28, 28)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.uint8)
        img = self.transform(img)
        if self.is_train:
            label = torch.tensor([self.labels[idx]], dtype=torch.long)
            return img, label
        return img
