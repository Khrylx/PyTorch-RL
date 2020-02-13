#!/bin/bash

xvfb-run --auto-servernum python drone_save_expert_traj.py --file=../assets/learned_models/Teste_ppo_209.p --env-name=Drone2 --max_expert_state_num=100000

