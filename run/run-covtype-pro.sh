#!/usr/bin/env bash

MAX_SAMPLES=10000000

mkdir -p logs

dataset_name=covtype
data_format=arff
kappa=0.4
ed=90

reuse_window_size=0
reuse_rate=0.18
lossy_window_size=100000000

# ProPEARL
../src/main.py --max_samples $MAX_SAMPLES --dataset_name $dataset_name --data_format $data_format \
    --cpp -t 60 -c 120 \
    -s --cd_kappa_threshold $kappa --edit_distance_threshold $ed \
    -p --proactive \
    --reuse_rate_upper_bound $reuse_rate \
    --reuse_rate_lower_bound $reuse_rate \
    --reuse_window_size $reuse_window_size \
    --lossy_window_size $lossy_window_size