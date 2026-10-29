import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class MNISTDataset(Dataset):
    def __init__(self, csv_file, is_train=True, train_mean=None, train_std=None):
        self.is_train = is_train
        self.data = pd.read_csv(csv_file)

        if is_train:
            self.labels = self.data["label"].values
            self.images = self.data.drop(columns=["label"]).values.reshape(-1, 28, 28)

            pre_images = self.images.astype(np.float32) / 255.0
            self.mean = np.mean(pre_images)
            self.std = np.std(pre_images)
        else:
            self.images = self.data.drop(columns=["label"]).values.reshape(-1, 28, 28)
            self.mean = train_mean
            self.std = train_std

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((self.mean,), (self.std,))]
        )

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img = self.image[idx].astype(np.uint8)
            img = self.transform(img)
            if is_train:
                return img, self.labels[idx]
            return img
