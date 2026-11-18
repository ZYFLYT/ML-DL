import torch
import pandas as pd
from torch.utils.data import Dataset
from preprocess import tokenize


class TweetDataset(Dataset):
    def __init__(self, df, vocab, max_len=30, is_train=True):
        self.df = df
        self.vocab = vocab
        self.max_len = max_len
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["text"]
        tokens = tokenize(text)
        ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]

        # 截断或填充 (真实推文有些极长，有些极短，必须统一到 max_len)
        if len(ids) < self.max_len:
            ids += [self.vocab["<PAD>"]] * (self.max_len - len(ids))
        else:
            ids = ids[: self.max_len]

        text_tensor = torch.tensor(ids, dtype=torch.long)

        if self.is_train:
            label = self.df.iloc[idx]["target"]
            label_tensor = torch.tensor(label, dtype=torch.float32)
            return text_tensor, label_tensor
        return text_tensor
