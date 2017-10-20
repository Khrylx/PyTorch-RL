import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.math import *


class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_num):
        super().__init__()
        self.affine1 = nn.Linear(state_dim, 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_head = nn.Linear(64, action_num)
        self.action_head.weight.data.mul_(0.1)
        self.action_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        action_prob = F.softmax(self.action_head(x))

        return action_prob

    def select_action(self, x):
        action_prob = self.forward(x)
        action = action_prob.multinomial()
        return action.data

    def get_kl(self, x):
        action_prob1 = self.forward(x)
        action_prob0 = Variable(action_prob1.data)
        kl = action_prob0 * (torch.log(action_prob0) - torch.log(action_prob1))
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_prob = self.forward(x)
        return action_prob.gather(1, actions.unsqueeze(1))

