import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import time

from dataset import MNISTDataset
from model import MyMLP


if __name__ == "__main__":
    #  自动检测本地 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前计算设备: {device}")
    if device.type == "cuda":
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")

    #  加载本地数据
    train_dataset = MNISTDataset("./data/train.csv", is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

    test_dataset = MNISTDataset("./data/test.csv", is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # 实例化模型、Loss、优化器
    model = MyMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    print("开始训练...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_dataset):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10000 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10000:.4f}"
                )
                running_loss = 0.0

    print(f"🎉 训练全部完成！耗时: {time.time() - start_time:.2f} 秒")

    print("开始预测测试集...")
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())

    submission = pd.DataFrame(
        {"ImageId": range(1, len(all_preds) + 1), "Label": all_preds}
    )
    submission.to_csv("submission.csv", index=False)
    print("submission.csv 已经生成在 MNIST-MLP 目录下！")
