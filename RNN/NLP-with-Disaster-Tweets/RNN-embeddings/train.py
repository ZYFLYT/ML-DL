import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from preprocess import build_vocab
from dataset import TweetDataset
from model import EmbeddingRNNClassifier

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"当前设备: {torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'}"
    )

    # 1. 读取数据
    train_df = pd.read_csv("../data/train.csv")
    test_df = pd.read_csv("../data/test.csv")
    train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)

    # 2. 构建词表
    vocab = build_vocab(train_data["text"].values, min_freq=2)
    print(f"词表大小: {len(vocab)}")

    # 3. DataLoader (优化：batch_size 从 64 提升到 128，max_len 用默认的 50)
    train_dataset = TweetDataset(train_data, vocab, max_len=50, is_train=True)
    val_dataset = TweetDataset(val_data, vocab, max_len=50, is_train=True)
    test_dataset = TweetDataset(test_df, vocab, max_len=50, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    # 4. 初始化 Embedding 模型
    model = EmbeddingRNNClassifier(
        vocab_size=len(vocab), embed_dim=128, hidden_dim=64
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. 训练
    epochs = 15
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x).squeeze(1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
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

    # 6. 生成提交文件
    print("生成提交文件")
    model.eval()
    all_preds_test = []
    with torch.no_grad():
        for batch_x in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x).squeeze(1)
            preds = torch.sigmoid(logits).cpu().numpy() > 0.5
            all_preds_test.extend(preds.astype(int))

    submission = pd.DataFrame({"id": test_df["id"], "target": all_preds_test})
    submission.to_csv("submission_embedding.csv", index=False)
