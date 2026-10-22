import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import time

from dataset import F1Dataset
from model import F1MLP

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前计算设备: {device}")
    if device.type == "cuda":
        print(torch.cuda.get_device_name())

    train_dataset = F1Dataset("./data/train.csv")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

    test_dataset = F1Dataset(
        "./data/test.csv",
        is_train=False,
        train_means=train_dataset.means,
        train_std=train_dataset.stds,
        feature_cols=train_dataset.feature_cols,
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=0)

    model = F1MLP().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 5
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (feature_tensor, labels) in enumerate(train_dataset):
            feature_tensor, labels = feature_tensor.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(feature_tensor)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10000 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10000:.4f}"
                )
                running_loss = 0.0

    print(f"训练全部完成！耗时: {time.time() - start_time:.2f} 秒")

    model.eval()
    all_probs = []

    with torch.no_grad():
        for feature_tensor in test_loader:
            feature_tensor = feature_tensor.to(device)
            outputs = model(feature_tensor)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.squeeze().cpu().numpy())

    test_raw = pd.read_csv("./data/test.csv")
    submission = pd.DataFrame(
        {
            "id": test_raw["id"],  # ⚠️ 确保列名是 'id'，看下 sample_submission.csv
            "PitStop": all_probs,  # ⚠️ 确保列名是 'PitStop'，看下 sample_submission.csv
        }
    )
    submission.to_csv("submission.csv", index=False)
    print("submission.csv 已经生成在 MNIST-MLP 目录下！")
