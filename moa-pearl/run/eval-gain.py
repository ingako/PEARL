#!/usr/bin/env python3

import os
import sys
import math
import pandas as pd
import numpy as np

def is_empty_file(fpath):
    return False if os.path.isfile(fpath) and os.path.getsize(fpath) > 0 else True


base_dir = os.getcwd()
generator = sys.argv[1]
seed = sys.argv[2]

cur_data_dir = f"{base_dir}/{generator}"

arf_output  = f"{cur_data_dir}/result-arf.csv"
arf_df = pd.read_csv(arf_output)
arf_acc = arf_df["classifications correct (percent)"]

gain_report_path = f"{cur_data_dir}/gain-report.txt"

print(f"evaluating {generator}...")
print("evaluating params...")

gain_report_out = open(gain_report_path, "w")
gain_report_out.write("#instances,lru_queue_size,perf_eval_window,kappa,ed,sarf-arf,parf-arf,parf-sarf\n")
param_strs = ["lru_queue_size", "performance_eval_window", "kappa", "edit_distance"]


def eval_pearl_output(cur_data_dir, param_values, gain_report_out):
    if len(param_values) != len(param_strs):
        # recurse
        params = [f for f in os.listdir(cur_data_dir) if os.path.isdir(os.path.join(cur_data_dir, f))]
        print(f"evaluating {params}...")
        for cur_param in params:
            param_values.append(cur_param)
            eval_pearl_output(f"{cur_data_dir}/{cur_param}", param_values, gain_report_out)
            param_values.pop()

    else:
        # sarf_output = f"{cur_data_dir}/result-{seed}.csv"
        sarf_output = f"{cur_data_dir}/result.csv"
        gain_output = f"{cur_data_dir}/gain.csv"

        with open(gain_output, "w") as out:

            if is_empty_file(sarf_output):
                return

            sarf_df = pd.read_csv(sarf_output)
            sarf_acc = sarf_df["classifications correct (percent)"]

            num_instances = sarf_df["learning evaluation instances"]

            out.write("#count,sarf-arf-gain,parf-arf-gain,parf-sarf-gain\n")

            end = min(int(sys.argv[3]), min(len(sarf_acc), len(arf_acc)))

            sarf_arf_gain = 0
            parf_arf_gain = 0
            parf_sarf_gain = 0

            for i in range(0, end):
                sarf_arf_gain += sarf_acc[i] - arf_acc[i]

                if i == (end - 1):
                    gain_report_out.write(f"{num_instances[i]},"
                                          f"{param_values[0]},{param_values[1]},"
                                          f"{param_values[2]},{param_values[3]},"
                                          f"{sarf_arf_gain}\n")

                out.write(f"{num_instances[i]},"
                          f"{sarf_arf_gain},"
                          f"{parf_arf_gain},"
                          f"{parf_sarf_gain}\n")

                out.flush()

eval_pearl_output(cur_data_dir, [], gain_report_out)
gain_report_out.close()
