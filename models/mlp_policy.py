import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(state_dim, 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x):
        x = F.tanh(self.affine1(x))
        x = F.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

