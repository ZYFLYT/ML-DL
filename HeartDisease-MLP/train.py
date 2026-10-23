import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from dataset import HDDataset
from model import HDModel

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(torch.cuda.get_device_name())

    full_data = pd.read_csv("./data/train.csv")

    train_df, val_df = train_test_split(
        full_data, test_size=0.2, random_state=42, stratify=full_data["Heart Disease"]
    )  # stratify=full_data["Heart Disease"] 保证切分后正负样本比例一致

    train_df.to_csv("./data/train_split.csv", index=False)
    val_df.to_csv("./data/val_split.csv", index=False)

    train_dataset = HDDataset("./data/train_split.csv")
    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=4
    )

    val_dataset = HDDataset(
        "./data/val_split.csv",
        is_train=False,
        train_means=train_dataset.means,
        train_stds=train_dataset.stds,
        feature_cols=train_dataset.feature_cols,
    )
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

    test_dataset = HDDataset(
        "./data/test.csv",
        is_train=False,
        train_means=train_dataset.means,
        train_stds=train_dataset.stds,
        feature_cols=train_dataset.feature_cols,
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    input_dim = train_dataset.feature_tensor.shape[1]

    model = HDModel(input_dim=input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (feature_tensor, labels_tensor) in enumerate(train_loader):
            labels_tensor = labels_tensor.unsqueeze(1)

            feature_tensor, labels_tensor = (
                feature_tensor.to(device),
                labels_tensor.to(device),
            )
            optimizer.zero_grad()
            outputs = model(feature_tensor)
            loss = criterion(outputs, labels_tensor)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 1000 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 1000:.4f}"
                )
                running_loss = 0.0

        # --- 验证阶段 (每个 Epoch 跑完后验证一次) ---
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for feature_tensor, labels_tensor in val_loader:
                feature_tensor = feature_tensor.to(device)
                outputs = model(feature_tensor)
                probs = torch.sigmoid(outputs).squeeze()

                # 收集预测概率和真实标签
                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(labels_tensor.cpu().numpy())

        # 计算验证集 AUC
        auc = roc_auc_score(val_targets, val_preds)
        print(f"Epoch [{epoch + 1}] 验证集 AUC 得分: {auc:.4f}")
        print("-" * 50)

    model.eval()
    all_probs = []

    with torch.no_grad():
        for feature_tensor in test_loader:
            feature_tensor = feature_tensor.to(device)
            outputs_test = model(feature_tensor)
            probs = torch.sigmoid(outputs_test)
            all_probs.extend(probs.squeeze().cpu().numpy())

    test_raw = pd.read_csv("./data/test.csv")
    submission = pd.DataFrame({"id": test_raw["id"], "Heart Disease": all_probs})

    submission.to_csv("submission.csv", index=False)
