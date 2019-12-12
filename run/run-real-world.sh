#!/usr/bin/env bash

MAX_SAMPLES=1000000 

mkdir -p logs

generators=(covtype pokerhand elec weatherAUS airlines)
kappa_vals=(0.4 0.0 0.7 0.1 0.3)
ed_vals=(90 90 120 120 90)

for i in ${!generators[@]} ; do
    generator=${generators[$i]}
    kappa=${kappa_vals[$i]}
    ed=${ed_vals[$i]}

    cmd_str="./main.py --max_samples $MAX_SAMPLES --generator $generator -s \
    	--cd_kappa_threshold $kappa --edit_distance_threshold $ed \
    	> logs/$generator-s-$kappa-$ed.out 2>&1 &"
    echo $cmd_str

    ../src/main.py --max_samples $MAX_SAMPLES --generator $generator -s \
    	--cd_kappa_threshold $kappa --edit_distance_threshold $ed \
    	> logs/$generator-s-$kappa-$ed.out 2>&1 &

done
