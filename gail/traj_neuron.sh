#!/bin/bash


env_reset_mode=Discretized_Uniform
ENV_NAME=Stochastic_Policy
SEED=1

xvfb-run --auto-servernum --server-num=1 python drone_save_expert_traj.py --file=../assets/learned_models/Teste_ppo_499.p \
--env-name=${ENV_NAME} --max_expert_state_num=100000 \
--env_reset_mode=${env_reset_mode} --seed=${SEED} #--deterministic

