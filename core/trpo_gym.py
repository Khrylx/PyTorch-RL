import argparse
import torch
import gym
from itertools import count
from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from torch.autograd import Variable

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch TRPO example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Hopper-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=15000, metavar='N',
                    help='minimal batch size per TRPO update (default: 15000)')
parser.add_argument('--max-iter-num', type=int, default=10000, metavar='N',
                    help='maximal number of main iterations (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make(args.env_name)
env.seed(args.seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

running_state = ZFilter((state_dim,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)

policy_net = Policy(state_dim, action_dim)
value_net = Value(state_dim)


def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state, volatile=True))
    action = torch.normal(action_mean, action_std)
    return action


def main_loop():
    """generate mutiliple trajectories that reaches the minimum batch_size"""
    for i_iter in count():
        memory = Memory()

        num_steps = 0
        reward_batch = 0
        num_episodes = 0

        while num_steps < args.min_batch_size:
            state = env.reset()
            state = running_state(state)

            for t in range(10000): # Don't infinite loop while learning
                action = select_action(state)
                action = action.data[0].numpy()
                next_state, reward, done, _ = env.step(action)
                reward_batch += reward
                next_state = running_state(next_state)

                mask = 0 if done else 1

                memory.push(state, action, mask, next_state, reward)

                if args.render:
                    env.render()
                if done:
                    break

                state = next_state

            # log stats
            num_steps += (t+1)
            num_episodes += 1

        reward_batch /= num_episodes
        batch = memory.sample()
        # update_params(batch)


main_loop()
