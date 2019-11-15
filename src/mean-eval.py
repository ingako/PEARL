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

def get_mean(eval_func, dir_prefix, file_path_gen):
    result_list = []
    for seed in range(10):
        output_file = file_path_gen(dir_prefix, seed)
        result_list.append(eval_func(output_file)[0])

    result_acc = sum(result_list) / len(result_list)
    result_std = stdev(result_list)
    return f'${result_acc:.2f}\pm{result_std:.2f}$'

def is_empty_file(fpath):
    return False if os.path.isfile(fpath) and os.path.getsize(fpath) > 0 else True

base_dir = os.getcwd()

dataset='agrawal-6-gradual'
kappa=0.2
ed=90
reuse_rate=0.6
reuse_window_size=0

results = []

cur_data_dir = f"{base_dir}/{dataset}"
gain_report_path = f"{cur_data_dir}/gain-report.txt"

# arf results
# arf_output = f'{cur_data_dir}/result-{seed}.csv'
file_path_gen = lambda dir_prefix, seed : f'{dir_prefix}/result-{seed}.csv'
arf_result = get_mean(get_acc, cur_data_dir, file_path_gen)
print(arf_result)
