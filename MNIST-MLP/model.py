import torch.nn as nn


class MyMLP(nn.Module):
    def __init__(self):
        super(MyMLP, self).__init__()
        # 输入层 784 (28x28)，隐藏层 128，输出层 10
        self.fc1 = nn.Linear(in_features=784, out_features=128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # 展平
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
