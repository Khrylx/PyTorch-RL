#!/bin/bash
FILE=/home/gabriel/repos/PyTorch-RL/checkpoint/name_Testing_GAILS_2_clip_0.2_minbatch_2048_lr_0.0003_optepochs_10_optbatchs_64_init_Discretized_Uniform_seed_42_GAIL/GAIL_Testing_GAILS_2_itr_49.p



python evaluate_drone.py --file=${FILE} --env_reset_mode=Discretized_Uniform --render
