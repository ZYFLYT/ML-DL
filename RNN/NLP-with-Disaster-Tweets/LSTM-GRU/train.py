import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import time  # 用于计算训练时间

from preprocess import build_vocab
from dataset import TweetDataset
from model import LSTMClassifier, GRUClassifier  # 导入两个模型

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"当前设备: {torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'}"
    )

    # 1. 数据加载
    train_df = pd.read_csv("../data/train.csv")
    test_df = pd.read_csv("../data/test.csv")
    train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)

    vocab = build_vocab(train_data["text"].values, min_freq=2, max_vocab_size=5000)
    print(f"词表大小: {len(vocab)}")

    train_dataset = TweetDataset(train_data, vocab, max_len=40, is_train=True)
    val_dataset = TweetDataset(val_data, vocab, max_len=40, is_train=True)
    test_dataset = TweetDataset(test_df, vocab, max_len=40, is_train=False)

    # Windows 下 num_workers=0 保平安
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # ==========================================
    # 核心开关：True 用 GRU，False 用 LSTM
    # ==========================================
    use_gru = True

    if use_gru:
        print("正在使用单向 GRU 模型")
        model = GRUClassifier(vocab_size=len(vocab)).to(device)
        model_name = "GRU"
    else:
        print("正在使用单向 LSTM 模型")
        model = LSTMClassifier(vocab_size=len(vocab)).to(device)
        model_name = "LSTM"

    # 2. 计算类别权重 (解决样本不平衡问题)
    num_pos = sum(train_data["target"])
    num_neg = len(train_data) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)
    print(f"正样本权重: {pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. 开始训练
    epochs = 5
    start_time = time.time()

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

    print(f"{model_name} 训练完成！总耗时: {time.time() - start_time:.2f} 秒")

    # 4. 生成提交文件
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
    filename = f"submission_{model_name.lower()}.csv"
    submission.to_csv(filename, index=False)
