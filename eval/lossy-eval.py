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
    return df["memory"].iloc[-1]

def get_time(output):
    f = open(output,'r')
    out = f.readlines()
    return float(out[0][:-1])

base_dir = os.getcwd()

dataset='agrawal-3'
kappa=0.2
ed=110
reuse_rate = 0.4
reuse_window_size=0
no_lossy_win = 100000000

# dataset='agrawal-gradual'
# kappa=0.2
# ed=110

# dataset='agrawal-6-gradual'
# kappa=0.2
# ed=90

# dataset = 'agrawal-6-shift-66'
# kappa = 0.1
# ed = 90
# reuse_rate = 0.6
# reuse_window_size = 600
# no_lossy_win = 100000000

no_lossy_results = []
no_lossy_kappa_results = []
time_results = []
mem_results = []

lossy_results = []
lossy_kappa_results = []
lossy_time_results = []
lossy_mem_results = []

cur_data_dir = f"{base_dir}/{dataset}"

for seed in range(10):

    # arf results
    arf_output = f'{cur_data_dir}/result-{seed}.csv'
    arf_acc_sum = get_acc_sum(arf_output)
    arf_kappa_sum = get_kappa_sum(arf_output)

    # pattern matching results
    pattern_matching_dir = f'{cur_data_dir}/k{kappa}-e{ed}/'
    # sarf_output = f'{pattern_matching_dir}/result-sarf-{seed}.csv'
    # sarf_acc = get_acc(arf_output)

    cur_reuse_param = f"{pattern_matching_dir}/" \
                      f"r{reuse_rate}-r{reuse_rate}-w{reuse_window_size}"

    # pearl without lossy counting
    no_lossy_output = f'{cur_reuse_param}/lossy-{no_lossy_win}/result-parf-{seed}.csv'
    no_lossy_gain = get_acc_sum(no_lossy_output) - arf_acc_sum
    no_lossy_results.append(no_lossy_gain)
    no_lossy_kappa_results.append(get_kappa_sum(no_lossy_output) - arf_kappa_sum)
    mem_results.append(get_mem(no_lossy_output))
    time_output = f'{cur_reuse_param}/lossy-{no_lossy_win}/time-parf-{seed}.log'
    time_results.append(get_time(time_output))

    # pearl with lossy counting
    lossy_params = \
            [f for f in os.listdir(cur_reuse_param) if os.path.isdir(os.path.join(cur_reuse_param, f))]

    max_lossy_gain = 0
    kappa_gain = 0
    mem = 0
    time = 0

    for lossy_param in lossy_params:
        lossy_output = f'{cur_reuse_param}/{lossy_param}/result-parf-{seed}.csv'
        if is_empty_file(lossy_output):
            continue

        cur_acc_gain = get_acc_sum(lossy_output) - arf_acc_sum
        if cur_acc_gain > max_lossy_gain:
            max_lossy_gain = cur_acc_gain
            kappa_gain = get_kappa_sum(lossy_output) - arf_kappa_sum
            mem = get_mem(lossy_output)
            time_output = f'{cur_reuse_param}/{lossy_param}/time-parf-{seed}.log'
            time = get_time(time_output)

    lossy_results.append(max_lossy_gain)
    lossy_kappa_results.append(kappa_gain)
    lossy_time_results.append(time)
    lossy_mem_results.append(mem)

def eval(results, div):
    result_strs = []
    for idx, result in enumerate(results):
        mean = sum(result) / len(result) / div[idx]
        std = stdev(result) / div[idx]
        result_strs.append(f'${mean:.2f}\pm{std:.2f}$')
    print(' & '.join(result_strs))

div = [1, 1, 1024, 60]
eval([no_lossy_results, no_lossy_kappa_results, mem_results, time_results], div)
eval([lossy_results, lossy_kappa_results, lossy_mem_results, lossy_time_results], div)
