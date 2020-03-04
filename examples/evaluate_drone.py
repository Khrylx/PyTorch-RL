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
from core.ppo import ppo_step_one_loss, ppo_step_two_losses

from core.common import estimate_advantages
from core.agent import Agent

from larocs_sim.envs.drone_env import DroneEnv
import csv


def terminate():
    try:
        env.shutdown();import sys; sys.exit(0)
    except:
        import sys; sys.exit(0)


parser = argparse.ArgumentParser(description='PyTorch PPO example')
parser.add_argument('--env_reset_mode', default="Discretized_Uniform",
                    help='name of the environment to run')
parser.add_argument('--file', metavar='G',
                    help='path of pre-trained model')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--max_timesteps', type=int, default=1000,
                    help='Maximum timesteps for evaluation (default: 1000)')
parser.add_argument('--render', type=int, default=0, choices = [0,1],
                    help='To render or no the env')
parser.add_argument('--reward', type=str, default='Normal',
                    help="Reward Function")
parser.add_argument('--state', type=str, default='New', choices = ['New', 'Old', 'New_Double', 'New_Double'],
                    help="State Space")



args = parser.parse_args()



if args.env_reset_mode=='False':
    args.env_reset_mode = False


dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cpu')



"""environment"""
if args.render==0:
    headless=True
else:
    headless=False
env = DroneEnv(random=args.env_reset_mode,seed=args.seed, headless = headless, state=args.state, reward_function_name=args.reward)


action_dim = env.action_space.shape[0]
state_dim = env.observation_space[0]
is_disc_action = len(env.action_space.shape) == 0
running_state = ZFilter((state_dim,), clip=5)
# running_reward = ZFilter((1,), demean=False, clip=10)
"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# env.seed(args.seed)


policy_net, value_net, running_state = pickle.load(open(args.file, "rb"))


print(running_state.rs.mean)
print(running_state.rs.std)
# for _ in range(10):
    # state=env.reset()
    # print(state)
    # print(running_state(state))
terminate()

policy_net.to(device)
value_net.to(device)


"""create agent"""
agent = Agent(env, policy_net, device, running_state=running_state, num_threads=1, mean_action=True)


batch, log = agent.collect_samples(args.max_timesteps)

print('R_min {0:.2f}\tR_max {1:.2f}\tR_avg {2:.2f}\tNum_episodes {3:.2f}'.format(log['min_reward'], log['max_reward'], log['avg_reward'], log['num_episodes']))


env.shutdown()