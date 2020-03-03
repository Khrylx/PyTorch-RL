import torch.nn as nn
import torch
from utils.math import *
from torch.distributions import Normal


class Policy_Tanh_Gaussian(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(128, 128), activation='tanh', log_std=0, \
                use_reparametrization=True,log_std_min=-10, log_std_max=2):
        super().__init__()
        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)


        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.use_reparametrization = use_reparametrization
        self.epsilon = 1e-6
    
    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_log_std = torch.clamp(action_log_std, self.log_std_min, self.log_std_max)
        action_std = torch.exp(action_log_std)


        return action_mean, action_log_std, action_std

    def select_action_stochastic(self, x):


        action_mean, _, action_std = self.forward(x)
        
        normal = Normal(action_mean, action_std)
        if self.use_reparametrization:
            z = normal.rsample() ## Add reparam trick?
            z.requires_grad_()
        else:
            z = normal.sample() ## Add reparam trick?

        action = torch.tanh(z).detach().cpu()#.numpy()

        return action
    
    def select_action_deterministic(self, x):
        action_mean, _, action_std = self.forward(x)
        action = torch.tanh(action_mean).detach().cpu()#.numpy()

        return action
   
    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std = self.forward(x)
        normal = Normal(action_mean, action_std)
        z = normal.sample() ## Add reparam trick?
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + self.epsilon) ##  -  np.log(self.action_range) See gist https://github.com/quantumiracle/SOTA-RL-Algorithms/blob/master/sac_v2.py
        # log_prob = normal.log_prob(z) - torch.log(torch.clamp(1 - action.pow(2), min=0,max=1) + epsilon) # nao precisa por causa do squase tanh
        log_prob = log_prob.sum(-1, keepdim=True)

       
        return log_prob, action_mean, action_std
    
    # def get_entropy(self,std):
        # return normal_entropy(std)

