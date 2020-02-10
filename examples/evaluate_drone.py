import argparse
import gym
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent

from larocs_sim.envs.drone_env import DroneEnv
import csv



parser = argparse.ArgumentParser(description='PyTorch PPO example')
parser.add_argument('--env_reset_mode', default="Discretized_Uniform",
                    help='name of the environment to run')
parser.add_argument('--file', metavar='G',
                    help='path of pre-trained model')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--max_timesteps', type=int, default=1000,
                    help='Maximum timesteps for evaluation (default: 1000)')
parser.add_argument('--render', action='store_true', default=True,
                    help='To render or no the env')



args = parser.parse_args()




dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cpu')



"""environment"""
# env = gym.make(args.env_name)
print('args.render = ', (not args.render))


# import sys; sys.exit(0)
env = DroneEnv(random=args.env_reset_mode,seed=args.seed, headless = not args.render)

state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
running_state = ZFilter((state_dim,), clip=5)
# running_reward = ZFilter((1,), demean=False, clip=10)
"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# env.seed(args.seed)


policy_net, value_net, running_state = pickle.load(open(args.file, "rb"))


policy_net.to(device)
value_net.to(device)


"""create agent"""
agent = Agent(env, policy_net, device, running_state=running_state, num_threads=1, mean_action=True)


batch, log = agent.collect_samples(args.max_timesteps)

print('R_min {0:.2f}\tR_max {1:.2f}\tR_avg {2:.2f}\tNum_episodes {3:.2f}'.format(log['min_reward'], log['max_reward'], log['avg_reward'], log['num_episodes']))


env.shutdown()