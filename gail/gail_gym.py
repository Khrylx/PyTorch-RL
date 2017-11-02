import argparse
import gym
import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from models.mlp_discriminator import Discriminator
from torch.autograd import Variable
from torch import nn
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent

Tensor = DoubleTensor
torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch GAIL example')
parser.add_argument('--env-name', default="Hopper-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--expert-traj-path', metavar='G',
                    help='path of the expert trajectories')
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
                    help='gae (default: 3e-4)')
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
action_dim = (1 if is_disc_action else env_dummy.action_space.shape[0])
ActionTensor = LongTensor if is_disc_action else DoubleTensor

"""define actor, critic and discrimiator"""
if is_disc_action:
    policy_net = DiscretePolicy(state_dim, env_dummy.action_space.n)
else:
    policy_net = Policy(state_dim, env_dummy.action_space.shape[0])
value_net = Value(state_dim)
discrim_net = Discriminator(state_dim + action_dim)
discrim_criterion = nn.BCELoss()
if use_gpu:
    policy_net = policy_net.cuda()
    value_net = value_net.cuda()
    discrim_net = discrim_net.cuda()
    discrim_criterion = discrim_criterion.cuda()

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.learning_rate)

# optimization epoch number and batch size for PPO
optim_epochs = 5
optim_batch_size = 64

# load trajectory
expert_traj, running_state = pickle.load(open(args.expert_traj_path, "rb"))


def expert_reward(state, action):
    state_action = Tensor(np.hstack([state, action]))
    return -math.log(discrim_net(Variable(state_action, volatile=True)).data.numpy()[0])


"""create agent"""
agent = Agent(env_factory, policy_net, custom_reward=expert_reward,
              running_state=running_state, render=args.render, num_threads=args.num_threads)


def update_params(batch, i_iter):
    states = Tensor(batch.state)
    actions = ActionTensor(batch.action)
    rewards = Tensor(batch.reward)
    masks = Tensor(batch.mask)
    values = value_net(Variable(states, volatile=True)).data
    fixed_log_probs = policy_net.get_log_prob(Variable(states, volatile=True), Variable(actions)).data

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, Tensor)

    lr_mult = max(1.0 - float(i_iter) / args.max_iter_num, 0)

    """perform mini-batch PPO update"""
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm.tolist())
        states, actions, returns, advantages, fixed_log_probs = \
            states[perm], actions[perm], returns[perm], advantages[perm], fixed_log_probs[perm]

        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]
            expert_state_actions_b = Tensor(expert_traj[np.random.choice(expert_traj.shape[0],
                                                                         states_b.shape[0], replace=False), :])

            """update discriminator"""
            for _ in range(3):
                g_o = discrim_net(Variable(torch.cat([states_b, actions_b], 1)))
                e_o = discrim_net(Variable(expert_state_actions_b))
                optimizer_discrim.zero_grad()
                discrim_loss = discrim_criterion(g_o, Variable(ones((states_b.shape[0], 1)))) + \
                    discrim_criterion(e_o, Variable(zeros((states_b.shape[0], 1))))
                discrim_loss.backward()
                optimizer_discrim.step()

            """update generator"""
            ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, states_b, actions_b, returns_b,
                     advantages_b, fixed_log_probs_b, lr_mult, args.learning_rate, args.clip_epsilon, args.l2_reg)


def main_loop():
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size)
        update_params(batch, i_iter)

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\texpert_R_avg {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], log['avg_c_reward'], log['avg_reward']))

        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            pickle.dump((policy_net, value_net), open('../assets/learned_models/{}_gail.p'.format(args.env_name), 'wb'))


main_loop()
