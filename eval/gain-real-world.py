#!/usr/bin/env python3

import os
import sys
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

base_dir = os.getcwd()

datasets = ["elec", "pokerhand", "covtype", "airlines", "weatherAUS", "bike"]

configs = {}
configs["elec"]= Config(kappa=0.7, ed=120)
configs["pokerhand"]= Config(kappa=0.0, ed=90)
configs["covtype"]= Config(kappa=0.4, ed=90, reuse_rate=0.3, reuse_window_size=300, lossy_window=400)
configs["airlines"]= Config(kappa=0.3, ed=90)
configs["weatherAUS"]= Config(kappa=0.1, ed=120)
configs["bike"]= Config(kappa=0.0, ed=120)

results = [[] for _ in range(len(datasets))]

for idx, dataset in enumerate(datasets):
    cur_data_dir = f"{base_dir}/{dataset}"
    gain_report_path = f"{cur_data_dir}/gain-report.txt"

    config = configs[dataset]

    arf_output = f'{cur_data_dir}/result-0.csv'
    ecpf_ht_output = f'{cur_data_dir}/result-ecpf-ht.csv'
    ecpf_arf_output = f'{cur_data_dir}/result-ecpf-arf.csv'

    pattern_matching_dir = f'{cur_data_dir}/k{config.kappa}-e{config.ed}/'
    sarf_output = f'{pattern_matching_dir}/result-sarf-0.csv'
    parf_output = f'{pattern_matching_dir}/' \
                  f'r{config.reuse_rate}-r{config.reuse_rate}-w{config.reuse_window_size}' \
                  f'/lossy-{config.lossy_window}/result-parf-0.csv'

    gain_output = f"{cur_data_dir}/gain-ecpf.csv"

    with open(gain_report_path, "w") as gain_report_out, open(gain_output, "w") as out:

        gain_report_out.write("#instances,ecpfht-arf,ecpfarf-arf,sarf-arf\n")

        for metric in ["accuracy", "kappa"]:
            arf_df = pd.read_csv(arf_output)
            arf_acc = arf_df[metric]

            sarf_df = pd.read_csv(sarf_output)
            sarf_acc = sarf_df[metric]

            moa_metric_header = "classifications correct (percent)" if metric == 'accuracy' else "Kappa Statistic (percent)"
            ecpf_ht_df = pd.read_csv(ecpf_ht_output)
            ecpf_ht_acc = ecpf_ht_df[moa_metric_header]

            ecpf_arf_df = pd.read_csv(ecpf_arf_output)
            ecpf_arf_acc = ecpf_arf_df[moa_metric_header]

            num_instances = sarf_df["count"]

            out.write("#count,ecpfht-arf,ecpfarf-arf,sarf-arf\n")

            end = min(len(ecpf_ht_acc), len(sarf_acc))

            ecpfht_arf_gain = 0
            ecpfarf_arf_gain = 0
            sarf_arf_gain = 0

            for i in range(0, end):
                arf_val = 1 if math.isnan(arf_acc[i]) else arf_acc[i]
                sarf_val = 1 if math.isnan(sarf_acc[i]) else sarf_acc[i]

                ecpfht_arf_gain += ecpf_ht_acc[i] - arf_val * 100
                ecpfarf_arf_gain += ecpf_arf_acc[i] - arf_val * 100
                sarf_arf_gain += sarf_val * 100 - arf_val * 100

                if i == (end - 1):
                    results[idx].extend([ecpfht_arf_gain, ecpfarf_arf_gain, sarf_arf_gain])
                    gain_report_out.write(f"{num_instances[i]},"
                                          f"{ecpfht_arf_gain},"
                                          f"{ecpfarf_arf_gain},"
                                          f"{sarf_arf_gain}\n")

                out.write(f"{num_instances[i]},"
                          f"{ecpfht_arf_gain},"
                          f"{ecpfarf_arf_gain},"
                          f"{sarf_arf_gain}\n")

                out.flush()

for result in results:
    print(" & ".join([str(v) for v in result]))
