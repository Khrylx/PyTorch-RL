#!/bin/bash

# REWARD_FUNCTION=Normal

SEED=42
# CUDA=True
ENV_NAME=Teste
NUM_EPOCHS=500
env_reset_mode=Discretized_Uniform

MODEL_INTERVAL=10

experiment_name=${PREFIX}_Reward_${REWARD_FUNCTION}_lr_${learning_rate}_bat_${batch_size}_net_${net_size}_buff_${buffer_size}_numsteps_${num_steps_until_train}_numtrainperstep_${num_trains_per_step}_before_${min_num_steps_before_training}_reset_${env_reset_mode}


export OMP_NUM_THREADS=1
xvfb-run --auto-servernum python drone_ppo.py --env-name=${ENV_NAME} --env_reset_mode=${env_reset_mode} --seed=${SEED} \
--max-iter-num=${NUM_EPOCHS} --save-model-interval=${MODEL_INTERVAL}
