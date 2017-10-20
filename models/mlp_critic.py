import torch.nn as nn
import torch.nn.functional as F


class Value(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.affine1 = nn.Linear(num_inputs, 128)
        self.affine2 = nn.Linear(128, 128)

        self.value_head = nn.Linear(128, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.affine1(x))
        x = F.tanh(self.affine2(x))

        value = self.value_head(x)
        return value
