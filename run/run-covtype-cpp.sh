#!/usr/bin/env bash

MAX_SAMPLES=10000000

mkdir -p logs

generator=covtype
kappa=0.4
ed=90

# ARF
../src/main.py --max_samples $MAX_SAMPLES --generator $generator --cpp -t 60

# PEARL with pattern matching only
../src/main.py --max_samples $MAX_SAMPLES --generator $generator --cpp -t 60 -s \
    --cd_kappa_threshold $kappa --edit_distance_threshold $ed -c 120
