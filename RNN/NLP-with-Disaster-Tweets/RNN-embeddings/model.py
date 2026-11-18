import torch.nn as nn


class EmbeddingRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):  # 优化：引入 embed_dim
        super(EmbeddingRNNClassifier, self).__init__()
        # 1. 词嵌入层：把 ID 映射成 128 维稠密向量
        # padding_idx=0 告诉模型 0 是 <PAD>，不更新它的向量
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0
        )

        # 2. RNN 的 input_size 不再是 vocab_size，而是 embed_dim
        self.rnn = nn.RNN(
            input_size=embed_dim, hidden_size=hidden_dim, batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x 形状: (batch_size, seq_len)
        # 通过 Embedding 变成 (batch_size, seq_len, embed_dim)
        x_embed = self.embedding(x)

        # 送进 RNN，out 形状: (batch_size, seq_len, hidden_dim)
        out, hidden = self.rnn(x_embed)

        last_out = out[:, -1, :]

        x = self.dropout(last_out)
        return self.fc(x)
