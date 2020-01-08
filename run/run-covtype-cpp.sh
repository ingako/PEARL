#!/usr/bin/env bash

MAX_SAMPLES=10000000

mkdir -p logs

dataset_name=covtype
data_format=arff
kappa=0.4
ed=90

# ARF
../src/main.py --max_samples $MAX_SAMPLES --dataset_name $dataset_name --data_format $data_format --cpp -t 60

# PEARL with pattern matching only
../src/main.py --max_samples $MAX_SAMPLES --dataset_name $dataset_name --data_format $data_format \
    --cpp -t 60 -s \
    --cd_kappa_threshold $kappa --edit_distance_threshold $ed -c 120
