import argparse
import gym
import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itertools import count
from utils import *
from torch.autograd import Variable

Tensor = DoubleTensor
torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='Save expert trajectory')
parser.add_argument('--env-name', default="Hopper-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',
                    help='name of the expert model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--max-expert-state-num', type=int, default=50000, metavar='N',
                    help='maximal number of main iterations (default: 50000)')
args = parser.parse_args()

env = gym.make(args.env_name)
env.seed(args.seed)
torch.manual_seed(args.seed)
if use_gpu:
    torch.cuda.manual_seed_all(args.seed)

is_disc_action = len(env.action_space.shape) == 0
state_dim = env.observation_space.shape[0]

policy_net, _, running_state = pickle.load(open(args.model_path, "rb"))
expert_traj = []


def main_loop():

    num_steps = 0

    for i_episode in count():

        state = env.reset()
        state = running_state(state)
        reward_episode = 0

        for t in range(10000):
            state_var = Variable(Tensor(state).unsqueeze(0),  volatile=True)
            # choose mean action
            action = policy_net(state_var)[0].data[0].cpu().numpy()
            # choose stochastic action
            # action = policy_net.select_action(state_var)[0].cpu().numpy()
            action = int(action) if is_disc_action else action.astype(np.float64)
            next_state, reward, done, _ = env.step(action)
            next_state = running_state(next_state)
            reward_episode += reward
            num_steps += 1

            expert_traj.append(np.hstack([state, action]))

            if args.render:
                env.render()
            if done or num_steps >= args.max_expert_state_num:
                break

            state = next_state

        print('Episode {}\t reward: {:.2f}'.format(i_episode, reward_episode))

        if num_steps >= args.max_expert_state_num:
            break


main_loop()
expert_traj = np.stack(expert_traj)
pickle.dump((expert_traj, running_state), open(os.path.join(assets_dir(), 'expert_traj/{}_expert_traj.p'.format(args.env_name)), 'wb'))
