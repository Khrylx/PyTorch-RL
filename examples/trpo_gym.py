import argparse
import gym
from itertools import count
from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
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
                    help='minimal batch size per TRPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=10000, metavar='N',
                    help='maximal number of main iterations (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make(args.env_name)
env.seed(args.seed)
torch.manual_seed(args.seed)
if use_gpu:
    torch.cuda.manual_seed_all(args.seed)

state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
ActionTensor = LongTensor if is_disc_action else DoubleTensor

running_state = ZFilter((state_dim,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)

"""define actor and critic"""
if is_disc_action:
    policy_net = DiscretePolicy(state_dim, env.action_space.n)
else:
    policy_net = Policy(state_dim, env.action_space.shape[0])
value_net = Value(state_dim)
if use_gpu:
    policy_net = policy_net.cuda()
    value_net = value_net.cuda()

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=0.01)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=0.01)


def update_params(batch):
    states = Tensor(batch.state)
    actions = ActionTensor(batch.action)
    rewards = Tensor(batch.reward)
    masks = Tensor(batch.mask)
    values = value_net(Variable(states, volatile=True)).data

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, Tensor)

    """perform TRPO update"""
    trpo_step(policy_net, value_net, optimizer_value, states, actions, returns, advantages, args.max_kl, args.damping, args.l2_reg)


def main_loop():
    """generate multiple trajectories that reach the minimum batch_size"""
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
                state_var = Variable(Tensor(state).unsqueeze(0))
                action = policy_net.select_action(state_var)[0].numpy()
                action = int(action) if is_disc_action else action.astype(np.float64)
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
