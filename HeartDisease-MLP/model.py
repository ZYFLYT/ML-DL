import torch.nn as nn


class HDModel(nn.Module):
    def __init__(self, input_dim):
        super(HDModel, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
