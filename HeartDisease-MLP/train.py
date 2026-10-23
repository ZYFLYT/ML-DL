import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import HDDataset
from model import HDModel

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(torch.cuda.get_device_name())

    train_dataset = HDDataset("./data/train.csv")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

    input_dim = train_dataset.feature_tensor.shape[1]

    test_dataset = HDDataset(
        "./data/test.csv",
        is_train=False,
        train_means=train_dataset.means,
        train_stds=train_dataset.stds,
        feature_cols=train_dataset.feature_cols,
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

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
