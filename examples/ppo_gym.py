import argparse
import gym
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from torch.autograd import Variable
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent

Tensor = DoubleTensor
torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch PPO example')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',
                    help='path of pre-trained model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=0, metavar='G',
                    help='log std for the policy (default: 0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='learning rate (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
args = parser.parse_args()


def env_factory(thread_id):
    env = gym.make(args.env_name)
    env.seed(args.seed + thread_id)
    return env


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if use_gpu:
    torch.cuda.manual_seed_all(args.seed)

env_dummy = env_factory(0)
state_dim = env_dummy.observation_space.shape[0]
is_disc_action = len(env_dummy.action_space.shape) == 0
ActionTensor = LongTensor if is_disc_action else DoubleTensor

running_state = ZFilter((state_dim,), clip=5)
# running_reward = ZFilter((1,), demean=False, clip=10)

"""define actor and critic"""
if args.model_path is None:
    if is_disc_action:
        policy_net = DiscretePolicy(state_dim, env_dummy.action_space.n)
    else:
        policy_net = Policy(state_dim, env_dummy.action_space.shape[0], log_std=args.log_std)
    value_net = Value(state_dim)
else:
    policy_net, value_net, running_state = pickle.load(open(args.model_path, "rb"))
if use_gpu:
    policy_net = policy_net.cuda()
    value_net = value_net.cuda()
del env_dummy

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)

# optimization epoch number and batch size for PPO
optim_epochs = 5
optim_batch_size = 4096

"""create agent"""
agent = Agent(env_factory, policy_net, running_state=running_state, render=args.render, num_threads=args.num_threads)


def update_params(batch, i_iter):
    states = torch.from_numpy(np.stack(batch.state))
    actions = torch.from_numpy(np.stack(batch.action))
    rewards = torch.from_numpy(np.stack(batch.reward))
    masks = torch.from_numpy(np.stack(batch.mask).astype(np.float64))
    if use_gpu:
        states, actions, rewards, masks = states.cuda(), actions.cuda(), rewards.cuda(), masks.cuda()
    values = value_net(Variable(states, volatile=True)).data
    fixed_log_probs = policy_net.get_log_prob(Variable(states, volatile=True), Variable(actions)).data

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, use_gpu)

    lr_mult = max(1.0 - float(i_iter) / args.max_iter_num, 0)

    """perform mini-batch PPO update"""
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm).cuda() if use_gpu else LongTensor(perm)

        states, actions, returns, advantages, fixed_log_probs = \
            states[perm], actions[perm], returns[perm], advantages[perm], fixed_log_probs[perm]

        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

            ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, states_b, actions_b, returns_b,
                     advantages_b, fixed_log_probs_b, lr_mult, args.learning_rate, args.clip_epsilon, args.l2_reg)


def main_loop():
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size)
        t0 = time.time()
        update_params(batch, i_iter)
        t1 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['min_reward'], log['max_reward'], log['avg_reward']))

        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            if use_gpu:
                policy_net.cpu(), value_net.cpu()
            pickle.dump((policy_net, value_net, running_state),
                        open(os.path.join(assets_dir(), 'learned_models/{}_ppo.p'.format(args.env_name)), 'wb'))
            if use_gpu:
                policy_net.cuda(), value_net.cuda()


main_loop()
