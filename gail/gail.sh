#!/bin/bash

xvfb-run --auto-servernum python drone_gail.py --expert-traj-path=../assets/expert_traj/Drone_expert_traj.p --max-iter-num=5000
