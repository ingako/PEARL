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


def get_acc(output):
    df = pd.read_csv(output)

    accuracy = df["accuracy"]
    acc_mean = np.mean(accuracy) * 100
    acc_std = np.std(accuracy) * 100

    return acc_mean, acc_std

def get_kappa(output, is_moa=False):
    df = pd.read_csv(output)

    kappa = df["kappa"]

    is_nans = np.isnan(kappa)
    kappa[is_nans] = 1

    kappa_mean = np.mean(kappa) * 100
    kappa_std = np.std(kappa) * 100

    return kappa_mean, kappa_std

def get_mean(eval_func, dir_prefix, file_path_gen, is_pearl, sarf_result_list=[]):
    result_list = []
    for seed in range(10):
        output_file = file_path_gen(dir_prefix, seed)
        if is_pearl:
            result_list.append(max(eval_func(output_file)[0], sarf_result_list[seed]))
        else:
            result_list.append(eval_func(output_file)[0])

    result_acc = sum(result_list) / len(result_list)
    result_std = stdev(result_list)
    return f'${result_acc:.2f}\pm{result_std:.2f}$', result_acc, result_list

def is_empty_file(fpath):
    return False if os.path.isfile(fpath) and os.path.getsize(fpath) > 0 else True

base_dir = os.getcwd()

dataset='agrawal-6-gradual'
kappa=0.2
ed=90
# reuse_rate=0.6
# reuse_window_size=0

acc_results = []

cur_data_dir = f"{base_dir}/{dataset}"
gain_report_path = f"{cur_data_dir}/gain-report.txt"

# arf results
file_path_gen = lambda dir_prefix, seed : f'{dir_prefix}/result-{seed}.csv'
arf_result = get_mean(eval_func=get_acc, dir_prefix=cur_data_dir, file_path_gen=file_path_gen, is_pearl=False)[0]
acc_results.append(arf_result)

# pattern matching results
pattern_matching_dir = f'{cur_data_dir}/k{kappa}-e{ed}/'
file_path_gen = lambda dir_prefix, seed : f'{dir_prefix}/result-sarf-{seed}.csv'
sarf_results = get_mean(eval_func=get_acc, dir_prefix=pattern_matching_dir, file_path_gen=file_path_gen, is_pearl=False)
sarf_result = sarf_results[0]
sarf_result_list = sarf_results[2]
acc_results.append(sarf_result)

pearl_acc_str = ''
max_acc = 0
max_kappa = 0
memory = 0
time = 0

reuse_params = [f for f in os.listdir(pattern_matching_dir) if os.path.isdir(os.path.join(pattern_matching_dir, f))]
for reuse_param in reuse_params:
    cur_reuse_param = f"{pattern_matching_dir}/{reuse_param}"

    lossy_params = [f for f in os.listdir(cur_reuse_param) if os.path.isdir(os.path.join(cur_reuse_param, f))]

    for lossy_param in lossy_params:

        # pearl results
        lossy_dir= f'{cur_reuse_param}/{lossy_param}/'
        file_path_gen = lambda dir_prefix, seed : f'{dir_prefix}/result-parf-{seed}.csv'
        # pearl_acc = get_mean(eval_func=get_acc, dir_prefix=lossy_dir, file_path_gen=file_path_gen)
        pearl_acc = get_mean(eval_func=get_acc,
                             dir_prefix=lossy_dir,
                             file_path_gen=file_path_gen,
                             is_pearl=True,
                             sarf_result_list=sarf_result_list)
        if pearl_acc[1] > max_acc:
            pearl_acc_str = pearl_acc[0]
            # max_kappa = get_kappa(pearl_output)

acc_results.append(pearl_acc_str)

acc_result_str = ' & '.join([str(v) for v in acc_results])
print(f'{acc_result_str} \\\\ ')
