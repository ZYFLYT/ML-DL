import torch
import torch.nn as nn
import torch.nn.functional as F


class OneHotRNNClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64):
        super(OneHotRNNClassifier, self).__init__()

        self.rnn = nn.RNN(
            input_size=vocab_size, hidden_size=hidden_dim, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x_onehot = F.one_hot(x, num_classes=self.rnn.input_size).float()

        # 输入张量: (batch_size, seq_len, vocab_size)
        # out 形状: (batch_size, seq_len, hidden_dim)
        out, hidden = self.rnn(x_onehot)

        # 只取最后一个时间步 (即 -1) 的隐藏状态，作为整句话的特征表示。
        # 形状变成: (batch_size, hidden_dim)
        last_out = out[:, -1, :]

        x = self.dropout(last_out)

        return self.fc(x)
