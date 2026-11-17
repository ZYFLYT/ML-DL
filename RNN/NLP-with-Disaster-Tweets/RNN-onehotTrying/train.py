import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from preprocess import build_vocab
from dataset import TweetDataset

# 导入我们手写的 One-hot RNN 模型
from model import OneHotRNNClassifier

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"当前设备: {torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'}"
    )

    # 1. 读取数据 (请确保路径正确)
    train_df = pd.read_csv("../data/train.csv")
    test_df = pd.read_csv("../data/test.csv")
    train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)

    # 2. 构建词表 (限制最大 3000 个词)
    vocab = build_vocab(train_data["text"].values, min_freq=2, max_vocab_size=3000)
    print(f"词表大小: {len(vocab)} (这决定了 One-hot 向量的长度)")

    # 3. DataLoader (由于 One-hot 比较大，batch_size 尽量小一点，这里用 64)
    train_dataset = TweetDataset(train_data, vocab, max_len=20, is_train=True)
    val_dataset = TweetDataset(val_data, vocab, max_len=20, is_train=True)
    test_dataset = TweetDataset(test_df, vocab, max_len=20, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # 4. 初始化模型
    # 注意：传入的 vocab_size 是动态计算出的词表长度（大概3000左右）
    model = OneHotRNNClassifier(vocab_size=len(vocab)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. 训练
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            # squeeze(1) 是为了把形状从 (batch, 1) 变成 (batch,)，匹配标签的形状
            outputs = model(batch_x).squeeze(1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证阶段
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x).squeeze(1)
                preds = torch.sigmoid(logits).cpu().numpy() > 0.5
                val_preds.extend(preds.astype(int))
                val_targets.extend(batch_y.numpy().astype(int))

        f1 = f1_score(val_targets, val_preds)
        print(
            f"Epoch [{epoch + 1}/{epochs}] Loss: {total_loss / len(train_loader):.4f} | Val F1: {f1:.4f}"
        )

    # 6. 生成 Kaggle 提交文件
    print("生成 Kaggle 提交文件...")
    model.eval()
    all_preds_test = []
    with torch.no_grad():
        for batch_x in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x).squeeze(1)
            preds = torch.sigmoid(logits).cpu().numpy() > 0.5
            all_preds_test.extend(preds.astype(int))

    submission = pd.DataFrame({"id": test_df["id"], "target": all_preds_test})
    submission.to_csv("submission_onehot.csv", index=False)
    print("✅ 完成！生成了 submission_onehot.csv")
