#!/usr/bin/env python3

import os
import subprocess
import math
import pandas as pd
import numpy as np

class Config:
    def __init__(self, kappa, ed, reuse_rate=0, reuse_window_size=0, lossy_window=0):
        self.kappa = kappa
        self.ed = ed
        self.reuse_rate = reuse_rate
        self.reuse_window_size = reuse_window_size
        self.lossy_window = lossy_window


def get_acc(output, is_moa=False):
    df = pd.read_csv(output)

    if is_moa:
        accuracy = df["classifications correct (percent)"]
        acc_mean = np.mean(accuracy)
        acc_stdev = np.std(accuracy)
    else:
        accuracy = df["accuracy"]
        acc_mean = np.mean(accuracy) * 100
        acc_stdev = np.std(accuracy) * 100

    return f'{acc_mean:.2f}\pm{acc_stdev:.2f}'

def get_kappa(output, is_moa=False):
    df = pd.read_csv(output)

    if is_moa:
        kappa = df["Kappa Statistic (percent)"]
        kappa_mean = np.mean(kappa)
        kappa_stdev = np.std(kappa)
    else:
        kappa = df["kappa"]

        is_nans = np.isnan(kappa)
        kappa[is_nans] = 1

        kappa_mean = np.mean(kappa) * 100
        kappa_stdev = np.std(kappa) * 100

    return f'{kappa_mean:.2f}\pm{kappa_stdev:.2f}'


base_dir = os.getcwd()
base_dir = f'{base_dir}/../data/'

datasets = ["elec", "pokerhand", "covtype", "airlines", "weatherAUS"]

configs = {}
configs["elec"]= Config(kappa=0.7, ed=120)
configs["pokerhand"]= Config(kappa=0.0, ed=90)
configs["covtype"]= Config(kappa=0.4, ed=90, reuse_rate=0.3, reuse_window_size=300, lossy_window=400)
configs["airlines"]= Config(kappa=0.3, ed=90)
configs["weatherAUS"]= Config(kappa=0.1, ed=120)

results = [[] for _ in range(len(datasets))]

for idx, dataset in enumerate(datasets):
    cur_data_dir = f"{base_dir}/{dataset}"
    gain_report_path = f"{cur_data_dir}/gain-report.txt"

    config = configs[dataset]

    # arf results
    arf_output = f'{cur_data_dir}/result-0.csv'

    # ecpf results
    ecpf_ht_output = f'{cur_data_dir}/result-ecpf-ht.csv'
    ecpf_arf_output = f'{cur_data_dir}/result-ecpf-arf.csv'

    # pearl results
    pattern_matching_dir = f'{cur_data_dir}/k{config.kappa}-e{config.ed}/'
    sarf_output = f'{pattern_matching_dir}/result-sarf-0.csv'

    results[idx].append(get_acc(arf_output, is_moa=False))
    results[idx].append(get_acc(ecpf_ht_output, is_moa=True))
    results[idx].append(get_acc(ecpf_arf_output, is_moa=True))
    results[idx].append(get_acc(sarf_output, is_moa=False))

    results[idx].append(get_kappa(arf_output, is_moa=False))
    results[idx].append(get_kappa(ecpf_ht_output, is_moa=True))
    results[idx].append(get_kappa(ecpf_arf_output, is_moa=True))
    results[idx].append(get_kappa(sarf_output, is_moa=False))

num_instances = ['45,312', '829,012', '581,012', '539,383', '142,193']
for idx, result in enumerate(results):
    result_str = ' & '.join([f'${v}$' for v in result])
    print(f'{datasets[idx]} & {num_instances[idx]} & {result_str} \\\\ ')
    print('\midrule')
