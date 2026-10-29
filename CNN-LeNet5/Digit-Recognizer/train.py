import torch
from torch.utils.data import DataLoader, random_split

from dataset import Dataset
from model import LeNet5

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"当前设备:{torch.cuda.get_device_name() if device.name == 'cuda' else 'CPU'}"
    )

    full_train_dataset = Dataset("/data/train.csv")
    test_dataset = Dataset(
        "/data/test.csv",
        is_train=False,
        train_mean=full_train_dataset.mean,
        train_std=full_train_dataset.std,
    )

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4
    )
