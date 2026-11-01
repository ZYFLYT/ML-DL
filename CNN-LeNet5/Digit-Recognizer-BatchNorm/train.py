import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, random_split

from dataset import MNISTDataset
from model import LeNet5

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"当前设备:{torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'}"
    )

    full_train_dataset = MNISTDataset("./data/train.csv")
    test_dataset = MNISTDataset(
        "./data/test.csv",
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
        train_dataset, batch_size=128, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    model = LeNet5(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 50:.4f}"
                )
                running_loss = 0.0

        # 验证
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, dim=1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch [{epoch + 1}] 验证集准确率: {acc * 100:.2f}")

    model.eval()
    all_preds_test = []

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, dim=1)
            all_preds_test.extend(pred.cpu().numpy())

    submission = pd.DataFrame(
        {"ImageId": range(1, len(all_preds_test) + 1), "Label": all_preds_test}
    )
    submission.to_csv("submission.csv", index=False)
