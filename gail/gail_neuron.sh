#!/bin/bash

SEED=42
env_reset_mode=Discretized_Uniform
MODEL_INTERVAL=50
log_interval=1

CLIP_EPSILON=0.2
MINBATCH=2048
lr=0.0003
OPTIM_EPOCHS=10
OPTIM_BATCHSIZE=64



ENV_NAME=Testing_Stochastic
EXPERT_TRAJ=../assets/expert_traj/Stochastic_Policy_expert_traj_stochasthic_itrs_100000.p
# EXPERT_TRAJ=../assets/expert_traj/Deterministic_Policy_expert_traj_deterministic_itrs_100000.p


xvfb-run --auto-servernum --server-num=1 python drone_gail.py --expert-traj-path=${EXPERT_TRAJ} \
--save-model-interval=${MODEL_INTERVAL} --log-interval=${log_interval} --env-name=${ENV_NAME} \
--env_reset_mode=${env_reset_mode} --seed=${SEED} --max-iter-num=5000 \
--save_path=name_${ENV_NAME}_clip_${CLIP_EPSILON}_minbatch_${MINBATCH}_lr_${lr}_optepochs_${OPTIM_EPOCHS}_optbatchs_${OPTIM_BATCHSIZE}_init_${env_reset_mode}_seed_${SEED}
