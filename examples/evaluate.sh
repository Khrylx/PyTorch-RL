#!/bin/bash
#FILE=/home/gabriel/repos/PyTorch-RL/checkpoint/name_PPO_final_clip_0.2_minbatch_1024_lr_0.0001_optepochs_15_optbatchs_256_init_False_seed_42/PPO_final_ppo_itr_2499.p 
FILE=/home/gabriel/repos/PyTorch-RL/checkpoint/name_NANPROBLEM_clip_0.2_minbatch_1024_lr_0.0001_optepochs_15_optbatchs_256_init_False_seed_42/NANPROBLEM_ppo_itr_1299.p

RESET_MODE=False
STATE=New_Double
python evaluate_drone.py --file=${FILE} --env_reset_mode=${RESET_MODE} --render=1 --state=${STATE}

