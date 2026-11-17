import re
import pandas as pd
from collections import Counter


def clean_text(text):
    """清洗真实推文数据"""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # 去除 URL
    text = re.sub(r"@\w+", "", text)  # 去除 @提及
    text = re.sub(r"[^a-z\s]", "", text)  # 只保留英文字母和空格（去除标点、数字）
    return text.strip()


def tokenize(text):
    """英文分词最简单的方式就是按空格切分"""
    return clean_text(text).split()


def build_vocab(texts, min_freq=2, max_vocab_size=3000):
    """构建词表，词频小于 min_freq 的判定为 <UNK>"""
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    most_common = counter.most_common(max_vocab_size)
    for word, freq in most_common:
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab
