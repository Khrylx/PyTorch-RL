import argparse
import gym
import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from larocs_sim.envs.drone_env import DroneEnv

from itertools import count
from utils import *




def collect_samples(deterministic=False, use_only_sucess_runs = True):
    #torch.randn(pid)
    num_steps = 0

    expert_traj = []
        

    for i_episode in count():

        print('count = ',i_episode)
        if i_episode % 5 == 0:
            env.restart=True
            env.reset()
            env.restart=False
        
        state = env.reset()
        state = running_state(state)
        reward_episode = 0
        last_traj = []
        for t in range(args.H):
            if use_only_sucess_runs:
                if len(expert_traj) >= np.floor(args.max_expert_state_num/args.H):
                    expert_traj = np.concatenate(expert_traj)

                    return expert_traj
            else:
                if num_steps >= args.max_expert_state_num:
                    expert_traj = np.stack(expert_traj)[:args.max_expert_state_num]

                    return expert_traj


            state_var = tensor(state).unsqueeze(0).to(dtype)
            with torch.no_grad():
                if deterministic:
                    action = policy_net(state_var)[0][0].numpy() ## Chose mean action
                else:
                    action = policy_net.select_action(state_var)[0].numpy() ## Chose mean action

            action = int(action) if is_disc_action else action.astype(np.float64)
            next_state, reward, done, _ = env.step(np.clip(action*100,a_min=-100, a_max=100))
            next_state = running_state(next_state)

            if use_only_sucess_runs:
                last_traj.append(np.hstack([state, action]))
            else:
                expert_traj.append(np.hstack([state, action]))

            mask = 0 if done else 1
            reward_episode += reward


            if done:
                
                break
        
            state = next_state
        if use_only_sucess_runs == True:
            if t == (args.H-1):
                expert_traj.append(last_traj)

        print("Episode Reward = {0:.2f}".format(reward_episode))
        num_steps += (t + 1)




parser = argparse.ArgumentParser(description='Save expert trajectory')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--file', metavar='G',
                    help='File location of the expert policy')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 42)')
parser.add_argument('--max_expert_state_num', type=int, default=1000, metavar='N',
                    help='maximal number of main iterations (default: 1000)')
parser.add_argument('--H', type=int, default=250, metavar='N',
                    help='Time horizon of each episode (default: 250)')
parser.add_argument('--env_reset_mode', default="Discretized_Uniform",
                    help='Type of env initialization')
parser.add_argument('--deterministic', action='store_true', default=False,
                    help='Get rollout with deterministic action render the environment')

args = parser.parse_args()


dtype = torch.float64
torch.set_default_dtype(dtype)
env = DroneEnv(random=args.env_reset_mode,seed=args.seed)

torch.manual_seed(args.seed)
is_disc_action = len(env.action_space.shape) == 0
state_dim = env.observation_space.shape[0]

policy_net, _, running_state = pickle.load(open(args.file, "rb"))
running_state.fix = True
expert_traj = []



expert_traj = collect_samples(deterministic = args.deterministic,  use_only_sucess_runs = True)

print(expert_traj.shape)
if args.deterministic:
    type_policy='deterministic'
else:
    type_policy='stochasthic'
    
pickle.dump((expert_traj, running_state), open(os.path.join(assets_dir(), 'expert_traj/{}_expert_traj_{}_itrs_{}.p'.format(\
                    args.env_name, type_policy, args.max_expert_state_num)), 'wb'))

env.shutdown()                    