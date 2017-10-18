import argparse
import gym
from itertools import count
from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from torch.autograd import Variable
from core.trpo import trpo_step
from core.common import estimate_advantages

Tensor = DoubleTensor
torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch TRPO example')
parser.add_argument('--env-name', default="Hopper-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
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
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per TRPO update (default: 10000)')
parser.add_argument('--max-iter-num', type=int, default=10000, metavar='N',
                    help='maximal number of main iterations (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make(args.env_name)
env.seed(args.seed)
torch.manual_seed(args.seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

running_state = ZFilter((state_dim,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)

policy_net = Policy(state_dim, action_dim)
value_net = Value(state_dim)
optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=0.01)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=0.01)


def update_params(batch):
    states = Tensor(batch.state)
    actions = Tensor(batch.action)
    rewards = Tensor(batch.reward)
    masks = Tensor(batch.mask)
    values = value_net(Variable(states))

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values.data, args.gamma, args.tau, Tensor)

    """update critic"""
    values_target = Variable(returns)
    value_loss = (values - values_target).pow(2).mean()
    # weight decay
    for param in value_net.parameters():
        value_loss += param.pow(2).sum() * args.l2_reg

    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    action_means, action_log_stds, action_stds = policy_net(Variable(states, volatile=True))
    fixed_log_probs = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data

    """define the loss function for TRPO"""
    def get_loss(volatile=False):
        action_means, action_log_stds, action_stds = policy_net(Variable(states, volatile=volatile))
        log_probs = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages) * torch.exp(log_probs - Variable(fixed_log_probs))
        return action_loss.mean()

    """define the procedure for calculating the policy's KL"""
    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)


def select_action(state):
    state = Tensor(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action


def main_loop():
    """generate mutiliple trajectories that reach the minimum batch_size"""
    for i_iter in count():
        memory = Memory()

        num_steps = 0
        reward_batch = 0
        num_episodes = 0

        while num_steps < args.min_batch_size:
            state = env.reset()
            state = running_state(state)
            reward_episode = 0

            for t in range(10000):
                action = select_action(state)
                action = action.data[0].numpy().astype(np.float64)
                next_state, reward, done, _ = env.step(action)
                reward_episode += reward
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
            reward_batch += reward_episode

        reward_batch /= num_episodes
        batch = memory.sample()
        update_params(batch)

        if i_iter % args.log_interval == 0:
            print('Iter {}\tLast reward: {}\tAverage reward {:.2f}'.format(
                i_iter, reward_episode, reward_batch))


main_loop()
