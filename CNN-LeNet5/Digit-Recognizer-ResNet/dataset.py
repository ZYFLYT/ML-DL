import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class LeafDataset(Dataset):
    def __init__(
        self, csv_file, img_dir, transform=None, is_train=True, label_mapping=None
    ):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train

        if self.is_train and label_mapping is None:
            self.classes = sorted(self.data["label"].unique())
            self.class_to_idx = {
                class_name: i for i, class_name in enumerate(self.classes)
            }
        else:
            self.class_to_idx = label_mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["image"]
        img_dir = os.path.join(self.img_dir, img_name)
        image = Image.open(img_dir).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_train:
            label_name = self.data.iloc[idx]["label"]
            label = self.class_to_idx[label_name]
            return image, label
        else:
            return image
