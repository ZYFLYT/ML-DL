import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import accuracy_score

from dataset import LeafDataset
from model import LeafResNet


# 封装一个独立地训练和验证函数
def run_epochs(
    model, train_loader, val_loader, optimizer, criterion, epochs, device, stage_name
):
    for epoch in range(epochs):
        # 训练
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(
                    f"[{stage_name}] Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 50:.4f}"
                )
                running_loss = 0.0

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                output = model(images)
                _, predict = torch.max(output, dim=1)
                all_preds.extend(predict.cpu().numpy())
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        print(
            f"===> [{stage_name}] Epoch [{epoch + 1}] 验证集准确率: {acc * 100:.2f}%\n"
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"当前设备:{torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'}"
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_csv_file = "./data/train.csv"
    test_csv_file = "./data/test.csv"
    img_dir = "./data/"

    full_data = pd.read_csv(train_csv_file)
    train_data = full_data.sample(frac=0.8, random_state=42)
    val_data = full_data.drop(train_data.index)

    train_dataset = LeafDataset(
        data=train_data,
        img_dir=img_dir,
        transform=train_transform,
        is_train=True,
    )
    val_dataset = LeafDataset(
        data=val_data,
        img_dir=img_dir,
        transform=val_transform,
        is_train=True,
        label_mapping=train_dataset.class_to_idx,
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()

    print("开始阶段 1:冻结主干网络,仅训练最后的全连接层")

    model = LeafResNet(num_classes=176, freeze_backbone=True).to(device)

    optimizer_stage1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
    )

    run_epochs(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer_stage1,
        criterion=criterion,
        epochs=3,
        device=device,
        stage_name="阶段1-冻结",
    )

    print("开始阶段 2:解冻 ResNet 的 layer4,使用小学习率进行微调 (Fine-tuning)")

    for param in model.backbone.layer4.parameters():
        param.requires_grad = True

    optimizer_stage2 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001
    )

    run_epochs(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer_stage2,
        criterion=criterion,
        epochs=2,
        device=device,
        stage_name="阶段2-微调",
    )

    # 输出预测
    model.eval()
    test_data = pd.read_csv(test_csv_file)
    test_dataset = LeafDataset(
        data=test_data,
        img_dir=img_dir,
        transform=val_transform,
        is_train=False,
        label_mapping=train_dataset.class_to_idx,
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    pre_result = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            output = model(images)
            _, predict = torch.max(output, dim=1)
            pre_result.extend(predict.cpu().numpy())

    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    predicted_labels = [idx_to_class[idx] for idx in pre_result]

    submission = pd.DataFrame({"image": test_data["image"], "label": predicted_labels})
    submission.to_csv("submission.csv", index=False)
