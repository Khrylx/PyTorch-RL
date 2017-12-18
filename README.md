# PyTorch implementation of reinforcement learning algorithms
I always try my best to make the code clean and more readable.
This repository contains:
1. policy gradient methods (TRPO, PPO, A2C)
2. [Generative Adversarial Imitation Learning (GAIL)](https://arxiv.org/pdf/1606.03476.pdf)

## Important notes
- To run mujoco environments, first install [mujoco-py](https://github.com/openai/mujoco-py) and my [modified version of gym](https://github.com/Khrylx/gym) which supports mujoco 1.50.
- If you have a GPU, I recommend setting the OMP_NUM_THREADS to 1 (PyTorch will create additional threads when performing computations which can damage the performance of multiprocessing. This problem is most serious with Linux, where multiprocessing can be even slower than a single thread):
```
export OMP_NUM_THREADS=1
```

## Features
* Support CUDA. (x10 faster than CPU implementation)
* Support discrete and continous action space.
* Support multiprocessing for agent to collect samples in multiple environments simultaneously. (x8 faster than single thread)
* Fast Fisher vector product calculation.
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
