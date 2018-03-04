import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.math import *


class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_num, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        self.is_disc_action = True
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_head = nn.Linear(last_dim, action_num)
        self.action_head.weight.data.mul_(0.1)
        self.action_head.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

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
        return torch.log(action_prob.gather(1, actions.unsqueeze(1)))

    def get_fim(self, x):
        action_prob = self.forward(x)
        M = action_prob.pow(-1).view(-1).data
        return M, action_prob, {}

