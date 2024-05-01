import torch.nn as nn


class MNISTModel(nn.Module):
    def __init__(self, n_input=28 * 28, n_output=10, n_hidden=128):
        super().__init__()

        self.l1 = nn.Linear(n_input, n_hidden)

        self.l2 = nn.Linear(n_hidden, n_output)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.l2(self.relu(self.l1(x)))
