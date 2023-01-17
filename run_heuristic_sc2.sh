#!/bin/bash

for i in `seq 1 10`;
do
    python3 src/main.py --config=heuristic --env-config=noisy_sc2_heuristic with env_args.map_name=1c3s5z
    python3 src/main.py --config=heuristic --env-config=noisy_sc2_heuristic with env_args.map_name=10m_vs_11m
    python3 src/main.py --config=heuristic --env-config=noisy_sc2_heuristic with env_args.map_name=3s5z
    python3 src/main.py --config=heuristic --env-config=noisy_sc2_heuristic with env_args.map_name=5m_vs_6m
    python3 src/main.py --config=heuristic --env-config=noisy_sc2_heuristic with env_args.map_name=3s_vs_5z
    python3 src/main.py --config=heuristic --env-config=noisy_sc2_heuristic with env_args.map_name=bane_vs_bane
done