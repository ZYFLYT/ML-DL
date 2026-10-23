import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class HDDataset(Dataset):
    def __init__(
        self,
        csv_file,
        is_train=True,
        train_means=None,
        train_stds=None,
        feature_cols=None,
    ):
        self.is_train = is_train
        data = pd.read_csv(csv_file)

        if self.is_train:
            # 修复：把 'Presence' 变成 1，'Absence' 变成 0
            if data["Heart Disease"].dtype == object:
                data["Heart Disease"] = data["Heart Disease"].map(
                    {"Presence": 1, "Absence": 0}
                )
            labels = data["Heart Disease"].values.astype(np.float32)
            feature_data = data.drop(["Heart Disease", "id"], errors="ignore")
        else:
            feature_data = data.drop(["id"], errors="ignore")

        numeric_cols = feature_data.select_dtypes(include=[np.number]).columns
        cat_cols = feature_data.select_dtypes(exclude=[np.number]).columns

        if len(cat_cols) > 0:
            onehot_data = pd.get_dummies(feature_data[cat_cols], dummy_na=False)
            num_data = feature_data[numeric_cols]
            feature_data = pd.concat([onehot_data, num_data], axis=1)
        else:
            feature_data = feature_data[numeric_cols]

        if self.is_train:
            self.feature_cols = feature_data.columns.to_list()
        else:
            feature_data = feature_data.reindex(columns=feature_cols, fill_value=0)

        if self.is_train:
            self.means = feature_data[numeric_cols].mean()
            self.stds = feature_data[numeric_cols].std()
            self.stds = self.stds.replace(0, 1.0).fillna(1.0)
            feature_data[numeric_cols] = (
                feature_data[numeric_cols] - self.means
            ) / self.stds
        else:
            feature_data[numeric_cols] = (
                feature_data[numeric_cols] - train_means[numeric_cols]
            ) / train_stds[numeric_cols]

        feature_data = feature_data.fillna(0)
        feature_data_float = feature_data.values.astype(np.float32)
        self.feature_tensor = torch.tensor(feature_data_float, dtype=torch.float32)
        if self.is_train:
            self.labels_tensor = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.feature_tensor)

    def __getitem__(self, idx):
        if self.is_train:
            return self.feature_tensor[idx], self.labels_tensor[idx]
        return self.feature_tensor[idx]
