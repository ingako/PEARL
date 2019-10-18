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

cur_data_dir = f"{base_dir}/{generator}"
mem_report_path = f"{cur_data_dir}/mem-report.txt"

print(f"evaluating memory consumption for {generator}...")
params = [f for f in os.listdir(cur_data_dir) if os.path.isdir(os.path.join(cur_data_dir, f))]
print("evaluating params...")
print(params)

with open(mem_report_path, "w") as mem_report_out:
    mem_report_out.write("param,reuse-param,lossy-win,#instances,sarf,parf,diff(MB)\n")

    for param in params:

        cur_param_dir = f"{cur_data_dir}/{param}"
        sarf_output = f"{cur_param_dir}/result-sarf-0.csv"
        sarf_time_output = f"{cur_param_dir}/time-sarf-0.log"

        reuse_params = [f for f in os.listdir(cur_param_dir) if os.path.isdir(os.path.join(cur_param_dir, f))]
        print(f"evaluating reuse params for {param}...")
        print(reuse_params)

        for reuse_param in reuse_params:

            cur_reuse_param = f"{cur_param_dir}/{reuse_param}"
            lossy_params = [f for f in os.listdir(cur_reuse_param) if os.path.isdir(os.path.join(cur_reuse_param, f))]

            for lossy_param in lossy_params:
                cur_lossy_param = f"{cur_reuse_param}/{lossy_param}"
                parf_output = f"{cur_lossy_param}/result-parf-0.csv"
                parf_time_output = f"{cur_lossy_param}/time-parf-0.log"

                sarf_df = pd.read_csv(sarf_output)
                sarf_mem = sarf_df["memory"]

                sarf_time_df = pd.read_csv(sarf_time_output, header=None)
                sarf_time = sarf_time_df[0][0]

                if is_empty_file(parf_output):
                    continue

                parf_df = pd.read_csv(parf_output)
                parf_mem = parf_df["memory"]

                parf_time_df = pd.read_csv(parf_time_output, header=None)
                parf_time = parf_time_df[0][0]

                diff_time = parf_time - sarf_time

                num_instances = parf_df["count"]

                end = min(len(sarf_mem), len(parf_mem)) - 1

                sarf_mem = sarf_mem[end] / 1024
                parf_mem = parf_mem[end] / 1024
                diff = parf_mem - sarf_mem

                mem_report_out.write(f"{param},{reuse_param},{lossy_param},{num_instances[end]},"
                                      f"{sarf_mem},"
                                      f"{parf_mem},"
                                      f"{diff},"
                                      f"{sarf_time/60},"
                                      f"{parf_time/60},"
                                      f"{diff_time/60}\n")
