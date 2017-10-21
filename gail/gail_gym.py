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

Tensor = DoubleTensor
torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch PPO example')
parser.add_argument('--env-name', default="Hopper-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--expert-traj-path', metavar='G',
                    help='path of the expert trajectories')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
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

env = gym.make(args.env_name)
env.seed(args.seed)
torch.manual_seed(args.seed)
if use_gpu:
    torch.cuda.manual_seed_all(args.seed)

state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
ActionTensor = LongTensor if is_disc_action else DoubleTensor

# running_state = ZFilter((state_dim,), clip=5)
# running_reward = ZFilter((1,), demean=False, clip=10)

"""define actor, critic and discrimiator"""
if is_disc_action:
    policy_net = DiscretePolicy(state_dim, env.action_space.n)
else:
    policy_net = Policy(state_dim, env.action_space.shape[0])
value_net = Value(state_dim)
discrim_net = Discriminator(state_dim + (1 if is_disc_action else env.action_space.shape[0]))
discrim_criterion = nn.BCELoss()
if use_gpu:
    policy_net = policy_net.cuda()
    value_net = value_net.cuda()
    discrim_net = discrim_net.cuda()
    discrim_criterion = discrim_criterion.cuda()

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
optimizer_discrim = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)

# optimization epoch number and batch size for PPO
optim_epochs = 5
optim_batch_size = 64

# load trajectory
expert_traj = pickle.load(open(args.expert_traj_path, "rb"))


def update_params(batch, i_iter):
    states = Tensor(batch.state)
    actions = ActionTensor(batch.action)
    rewards = Tensor(batch.reward)
    masks = Tensor(batch.mask)
    values = value_net(Variable(states, volatile=True)).data
    fixed_log_probs = policy_net.get_log_prob(Variable(states, volatile=True), Variable(actions)).data
    expert_state_actions = Tensor(expert_traj[np.random.choice(expert_traj.shape[0], states.shape[0], replace=False), :])

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, Tensor)

    lr_mult = max(1.0 - float(i_iter) / args.max_iter_num, 0)

    """perform mini-batch PPO update"""
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm.tolist())
        states, actions, returns, advantages, fixed_log_probs, expert_state_actions = \
            states[perm], actions[perm], returns[perm], advantages[perm], fixed_log_probs[perm], expert_state_actions[perm]

        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b, expert_state_actions_b = \
                states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind], expert_state_actions[ind]

            """update discriminator"""
            for _ in range(1):
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
    """generate multiple trajectories that reach the minimum batch_size"""
    for i_iter in range(args.max_iter_num):
        memory = Memory()

        num_steps = 0
        true_reward_batch = 0
        expert_reward_batch = 0
        num_episodes = 0

        while num_steps < args.min_batch_size:
            state = env.reset()
            # state = running_state(state)
            true_reward_episode = 0
            expert_reward_episode = 0

            for t in range(10000):
                state_var = Variable(Tensor(state).unsqueeze(0), volatile=True)
                action = policy_net.select_action(state_var)[0].cpu().numpy()
                action = int(action) if is_disc_action else action.astype(np.float64)
                next_state, true_reward, done, _ = env.step(action)
                true_reward_episode += true_reward
                # next_state = running_state(next_state)

                state_action = Tensor(np.hstack([state, action]))
                expert_reward = -math.log(discrim_net(Variable(state_action, volatile=True)).data.numpy()[0])
                expert_reward_episode += expert_reward

                mask = 0 if done else 1

                memory.push(state, action, mask, next_state, expert_reward)

                if args.render:
                    env.render()
                if done:
                    break

                state = next_state

            # log stats
            num_steps += (t+1)
            num_episodes += 1
            true_reward_batch += true_reward_episode
            expert_reward_batch += expert_reward_episode

        true_reward_batch /= num_episodes
        expert_reward_batch /= num_episodes
        batch = memory.sample()
        update_params(batch, i_iter)

        if i_iter % args.log_interval == 0:
            print('Iter {}\t  Average expert reward: {:.2f}\t  Average true reward {:.2f}'.format(
                i_iter, expert_reward_batch, true_reward_batch))

        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            pickle.dump((policy_net, value_net), open('../assets/learned_models/{}_ppo.p'.format(args.env_name), 'wb'))


main_loop()
