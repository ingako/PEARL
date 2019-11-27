#!/usr/bin/env python3

import os
import subprocess
import math
import pandas as pd
import numpy as np
from statistics import stdev

class Config:
    def __init__(self, kappa, ed, reuse_rate=0, reuse_window_size=0, lossy_window=0):
        self.kappa = kappa
        self.ed = ed
        self.reuse_rate = reuse_rate
        self.reuse_window_size = reuse_window_size
        self.lossy_window = lossy_window


def is_empty_file(fpath):
    return False if os.path.isfile(fpath) and os.path.getsize(fpath) > 0 else True

def get_acc(output):
    df = pd.read_csv(output)

    accuracy = df["accuracy"]
    acc_mean = np.mean(accuracy) * 100
    acc_std = np.std(accuracy) * 100

    return acc_mean # , acc_std

def get_kappa(output, is_moa=False):
    df = pd.read_csv(output)

    kappa = df["kappa"]

    is_nans = np.isnan(kappa)
    kappa[is_nans] = 1

    kappa_mean = np.mean(kappa) * 100
    kappa_std = np.std(kappa) * 100

    return kappa_mean # , kappa_std

def get_acc_sum(output):
    df = pd.read_csv(output)
    acc_sum = df["accuracy"].sum() * 100
    return acc_sum

def get_kappa_sum(output, is_moa=False):
    df = pd.read_csv(output)

    kappa = df["kappa"]
    is_nans = np.isnan(kappa)
    kappa[is_nans] = 1

    kappa_sum = kappa.sum() * 100

    return kappa_sum

def get_mem(output):
    df = pd.read_csv(output)
    return df["memory"].iloc[-1] / 1024

def get_time(output):
    f = open(output,'r')
    out = f.readlines()
    return float(out[0][:-1]) / 60

base_dir = os.getcwd()

dataset='agrawal-3'
kappa=0.2
ed=110

# dataset='agrawal-gradual'
# kappa=0.2
# ed=110

# dataset='agrawal-6-gradual'
# kappa=0.2
# ed=90

cur_data_dir = f"{base_dir}/{dataset}"

# arf results
arf_acc_results = []
arf_kappa_results = []
arf_time_results = []
arf_mem_results = []
arf_acc_sum_results = []
arf_kappa_sum_results = []

for seed in range(10):
    arf_output = f'{cur_data_dir}/result-{seed}.csv'

    arf_acc_results.append(get_acc(arf_output))
    arf_acc_sum_results.append(get_acc_sum(arf_output))

    arf_kappa_results.append(get_kappa(arf_output))
    arf_kappa_sum_results.append(get_kappa_sum(arf_output))

    arf_time_output = f'{cur_data_dir}/time-{seed}.log'
    if is_empty_file(arf_time_output):
        continue
    arf_time_results.append(get_time(arf_time_output))

# pattern matching results
sarf_acc_results = []
sarf_kappa_results = []
sarf_time_results = []
sarf_mem_results = []
sarf_acc_gain_results = []
sarf_kappa_gain_results = []

pattern_matching_dir = f'{cur_data_dir}/k{kappa}-e{ed}/'
for seed in range(10):
    sarf_output = f'{pattern_matching_dir}/result-sarf-{seed}.csv'

    sarf_mem_results.append(get_mem(sarf_output))

    sarf_acc_results.append(get_acc(sarf_output))
    sarf_acc_gain_results.append(get_acc_sum(sarf_output) - arf_acc_sum_results[seed])

    sarf_kappa_results.append(get_kappa(sarf_output))
    sarf_kappa_gain_results.append(get_kappa_sum(sarf_output) - arf_kappa_sum_results[seed])

    sarf_time_output = f'{pattern_matching_dir}/time-sarf-{seed}.log'
    if is_empty_file(sarf_time_output):
        continue
    sarf_time_results.append(get_time(sarf_time_output))

# pearl results
pearl_acc_results = []
pearl_kappa_results = []
pearl_time_results = []
pearl_mem_results = []
pearl_acc_gain_results = []
pearl_kappa_gain_results = []

for seed in range(10):
    acc = 0
    kappa = 0

    max_acc_gain = -1
    max_kappa_gain = 0

    mem = 0
    time = 0

    cur_param_dir = ''
    reuse_params = [f for f in os.listdir(pattern_matching_dir) if os.path.isdir(os.path.join(pattern_matching_dir, f))]

    for reuse_param in reuse_params:

        cur_reuse_param = f"{pattern_matching_dir}/{reuse_param}"

        lossy_params = \
                [f for f in os.listdir(cur_reuse_param) if os.path.isdir(os.path.join(cur_reuse_param, f))]

        for lossy_param in lossy_params:
            lossy_output = f'{cur_reuse_param}/{lossy_param}/result-parf-{seed}.csv'
            if is_empty_file(lossy_output):
                print(f'file does not exist: {lossy_output}')
                continue

            cur_acc_gain = get_acc_sum(lossy_output) - arf_acc_sum_results[seed]
            if max_acc_gain < cur_acc_gain:
                max_acc_gain = cur_acc_gain
                acc = get_acc(lossy_output)

                kappa = get_kappa(lossy_output)
                max_kappa_gain = get_kappa_sum(lossy_output) - arf_kappa_sum_results[seed]

                mem = get_mem(lossy_output)

                time_output = f'{cur_reuse_param}/{lossy_param}/time-parf-{seed}.log'
                if is_empty_file(time_output):
                    continue
                time = get_time(time_output)

    if max_acc_gain < sarf_acc_gain_results[seed]:
        acc = sarf_acc_results[seed]
        max_acc_gain = sarf_acc_gain_results[seed]
        kappa = sarf_kappa_results[seed]
        max_kappa_gain = sarf_kappa_gain_results[seed]
        mem = sarf_mem_results[seed]

        # time_output = f'{cur_reuse_param}/{lossy_param}/time-parf-{seed}.log'
        # if is_empty_file(time_output):
        #     continue
        time = sarf_time_results[seed]

    pearl_acc_results.append(acc)
    pearl_kappa_results.append(kappa)
    pearl_time_results.append(time)
    pearl_mem_results.append(mem)

    pearl_acc_gain_results.append(max_acc_gain)
    pearl_kappa_gain_results.append(max_kappa_gain)


def eval_mean(results):
    result_strs = []
    for result in results:
        mean = sum(result) / len(result)
        std = stdev(result)
        result_strs.append(f'${mean:.2f}\pm{std:.2f}$')
    print(' & '.join(result_strs))

print('===============results===============')
print('accuracy & kappa avg.')
eval_mean([arf_acc_results, sarf_acc_results, pearl_acc_results,
          arf_kappa_results, sarf_kappa_results, pearl_kappa_results])
print('\n')

print('accuracy & kappa gain')
eval_mean([sarf_acc_gain_results, pearl_acc_gain_results,
          sarf_kappa_gain_results, pearl_kappa_gain_results])
print('\n')

print('memory & runtime')
eval_mean([sarf_mem_results, pearl_mem_results,
          arf_time_results, sarf_time_results, pearl_time_results])
print('\n')
