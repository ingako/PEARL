#!/usr/bin/env bash

MAX_SAMPLES=400001

mkdir -p logs

generator=agrawal-3

kappa=0.2
ed=110
reuse_window_size=0
reuse_rate=0.4

for seed in {0..9} ; do
	./main.py --max_samples $MAX_SAMPLES --generator $generator \
		--generator_seed $seed \
		> logs/$generator.out 2>&1 &

	# sarf
	./main.py --max_samples $MAX_SAMPLES --generator $generator -s \
		--cd_kappa_threshold $kappa --edit_distance_threshold $ed \
		--generator_seed $seed \
		> logs/$generator-s-$kappa-$ed.out 2>&1 &

	# pearl without lossy counting
	./main.py --max_samples $MAX_SAMPLES --generator $generator -p \
		--reuse_rate_upper_bound $reuse_rate \
		--reuse_rate_lower_bound $reuse_rate \
		--cd_kappa_threshold $kappa --edit_distance_threshold $ed \
		--generator_seed $seed \
		--reuse_window_size $reuse_window_size \
		--lossy_window_size 100000000 \
		> logs/$generator-p-$kappa-$ed-$reuse_rate-$lossy_window_size.out 2>&1 &

	# pearl with lossy counting
	for ((lossy_window_size=160;lossy_window_size<1360;lossy_window_size+=60)); do
		./main.py --max_samples $MAX_SAMPLES --generator $generator -p \
			--reuse_rate_upper_bound $reuse_rate \
			--reuse_rate_lower_bound $reuse_rate \
			--cd_kappa_threshold $kappa --edit_distance_threshold $ed \
			--generator_seed $seed \
			--reuse_window_size $reuse_window_size \
			--lossy_window_size $lossy_window_size \
			> logs/$generator-p-$kappa-$ed-$reuse_rate-$lossy_window_size.out 2>&1 &
	done

	# if ! ((seed % 2 - 1)) && ((seed > 0)) ; then
	# 	wait
	# fi
	wait
done
