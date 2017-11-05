# PyTorch implementation of reinforcement learning algorithms
I always try my best to make the code clean and more readable.
This repository contains:
1. policy gradient methods (TRPO, PPO, A2C)
2. [Generative Adversarial Imitation Learning (GAIL)](https://arxiv.org/pdf/1606.03476.pdf)

## Features
* Support CUDA.
* Support discrete and continous action space.
* Support multiprocessing for agent to collect samples in multiple environments simultaneously.

## Policy gradient methods
* [Trust Region Policy Optimization (TRPO)](https://arxiv.org/pdf/1502.05477.pdf) -> [examples/trpo_gym.py](https://github.com/Khrylx/PyTorch-RL/blob/master/examples/trpo_gym.py)
* [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf) -> [examples/ppo_gym.py](https://github.com/Khrylx/PyTorch-RL/blob/master/examples/ppo_gym.py)
* [Synchronous A3C (A2C)](https://arxiv.org/pdf/1602.01783.pdf) -> [examples/a2c_gym.py](https://github.com/Khrylx/PyTorch-RL/blob/master/examples/a2c_gym.py)

### Example
* python examples/ppo_gym.py --env-name Hopper-v1

### Reference
* [ikostrikov/pytorch-trpo](https://github.com/ikostrikov/pytorch-trpo)
* [openai/baselines](https://github.com/openai/baselines)


## Generative Adversarial Imitation Learning (GAIL)
### To save trajectory
* python gail/save_expert_traj.py --model-path assets/expert_traj/Hopper-v1_ppo.p
### To do imitation learning
* python gail/gail_gym.py --env-name Hopper-v1 --expert-traj-path assets/expert_traj/Hopper-v1_expert_traj.p
