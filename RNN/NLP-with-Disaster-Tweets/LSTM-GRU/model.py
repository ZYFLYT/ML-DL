import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False,
        )

        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x_embed = self.embedding(x)

        # LSTM 返回 (out, (hidden, cell))
        out, (hidden, cell) = self.lstm(x_embed)

        # 掩码平均池化，过滤PAD位置
        mask = (x != 0).unsqueeze(-1).float()
        out_mask = out * mask
        sum_out = out_mask.sum(dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)
        avg_out = sum_mask / sum_out

        x = self.dropout(avg_out)
        return self.fc(x)


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False,
        )

        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x_embed = self.embedding(x)

        # GRU 只返回 (out, hidden)，没有 cell 状态
        out, hidden = self.gru(x_embed)

        # 掩码平均池化
        mask = (x != 0).unsqueeze(-1).float()
        out_masked = out * mask
        sum_out = out_masked.sum(dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)
        avg_out = sum_out / sum_mask

        x = self.dropout(avg_out)
        return self.fc(x)
