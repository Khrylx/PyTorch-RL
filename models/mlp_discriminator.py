import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.affine1 = nn.Linear(num_inputs, 128)
        self.affine2 = nn.Linear(128, 128)

        self.logic = nn.Linear(128, 1)
        self.logic.weight.data.mul_(0.1)
        self.logic.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.affine1(x))
        x = F.tanh(self.affine2(x))

        prob = F.sigmoid(self.logic(x))
        return prob
