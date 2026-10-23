import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class F1Dataset(Dataset):
    def __init__(
        self,
        csv_file,
        is_train=True,
        train_means=None,
        train_std=None,
        feature_cols=None,
    ):
        self.is_train = is_train
        data = pd.read_csv(csv_file)

        if self.is_train:
            self.labels = data["PitStop"].values.astype(np.float32)
            feature_data = data.drop(columns=["id", "PitStop"], errors="ignore")
        else:
            feature_data = data.drop(columns=["id", "PitStop"], errors="ignore")

        # 分别出原本的数据列，做标准化，文字的做one-hot
        numeric_cols = feature_data.select_dtypes(include=[np.number]).columns
        cat_cols = feature_data.select_dtypes(exclude=[np.number]).columns

        cat_dummy = pd.get_dummies(feature_data[cat_cols], dummy_na=False)
        num_data = feature_data[numeric_cols]

        feature_data = pd.concat([cat_dummy, num_data], axis=1)

        if self.is_train:
            self.feature_cols = feature_data.columns.tolist()
        else:
            feature_data = feature_data.reindex(columns=feature_cols, fill_value=0)

        if self.is_train:
            # 计算每一列的均值和标准差
            self.means = feature_data[numeric_cols].mean()
            self.stds = feature_data[numeric_cols].std()
            # 防坑：如果某列的所有值都一样（比如全是0），标准差算出来就是 NaN
            # 把标准差为 0 或 NaN 的地方替换成 1，防止下面做除法时报错
            self.stds = self.stds.replace(0, 1.0).fillna(1.0)
            feature_data[numeric_cols] = (
                feature_data[numeric_cols] - self.means
            ) / self.stds
        else:
            feature_data[numeric_cols] = (
                feature_data[numeric_cols] - train_means[numeric_cols]
            ) / train_std[numeric_cols]

        feature_data = feature_data.fillna(0)
        feature_data_float = feature_data.values.astype(np.float32)
        self.feature_tensor = torch.tensor(feature_data_float, dtype=torch.float32)

        if self.is_train:
            self.labels_tensor = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.feature_tensor)

    def __getitem__(self, idx):
        if self.is_train:
            return self.feature_tensor[idx], self.labels_tensor[idx]
        return self.feature_tensor[idx]
