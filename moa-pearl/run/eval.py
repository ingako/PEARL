#!/usr/bin/env python

import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["backend"] = "Qt4Agg"
plt.rcParams["figure.figsize"] = (20, 10)

@dataclass
class Params:
    kappa: float = 0.0
    ed: int = 100
    reuse_window: int = 0
    reuse_rate: float = 0.18
    lossy_window: int = 100000000

@dataclass
class Metric:
    acc_mean: float
    acc_std: float
    acc_gain: float
    kappa_gain: float
    runtime: float

dataset = sys.argv[1]
if dataset == "covtype":
    params = Params(
            kappa=0.4,
            ed=90,
            reuse_window=0,
            reuse_rate=0.18,
            lossy_window=100000000)

elif dataset == "sensor":
    params = Params(
            kappa=0.0,
            ed=100,
            reuse_window=0,
            reuse_rate=0.18,
            lossy_window=100000000)

elif dataset == "pokerhand":
    params = Params(
            kappa=0.0,
            ed=100,
            reuse_window=0,
            reuse_rate=0.18,
            lossy_window=100000000)

arf_path = f"{dataset}/result-arf.csv"
diversity_path = f"{dataset}/result-diversity-pool.csv"
ecpf_ht_path = f"{dataset}/result-ecpf-ht.csv"
ecpf_arf_path = f"{dataset}/result-ecpf-arf.csv"
pearl_path = f"{dataset}/{params.kappa}/{params.ed}/result.csv"
# pearl_path = f"{dataset}/{kappa}/{ed}/{reuse_rate}/{reuse_window}/{lossy_window}/result.csv"

arf = pd.read_csv(arf_path, index_col=0)
diversity = pd.read_csv(diversity_path, index_col=0)
ecpf_ht = pd.read_csv(ecpf_ht_path, index_col=0)
ecpf_arf = pd.read_csv(ecpf_arf_path, index_col=0)
pearl = pd.read_csv(pearl_path, index_col=0)


def get_metric(benchmark, arf):
    acc_mean = benchmark["evaluation time (cpu seconds)"].mean()
    acc_std = benchmark["evaluation time (cpu seconds)"].std()
    acc_gain = sum(benchmark["classifications correct (percent)"]) - sum(arf["classifications correct (percent)"])
    kappa_gain = sum(benchmark["Kappa Statistic (percent)"]) - sum(arf["Kappa Statistic (percent)"])
    runtime  = benchmark["evaluation time (cpu seconds)"].iloc[-1]
    return f"{acc_mean:.2f} \pm {acc_std:.2f} & {int(acc_gain)} & {int(kappa_gain)} & {runtime:.2f} \\\\"


arf_runtime  = arf["evaluation time (cpu seconds)"].iloc[-1]
print(f"- & - & {arf_runtime}")
print(get_metric(diversity, arf))
print(get_metric(ecpf_ht, arf))
print(get_metric(ecpf_arf, arf))
print(get_metric(pearl, arf))


plt.plot(arf["classifications correct (percent)"], label="ARF")
plt.plot(diversity["classifications correct (percent)"], label="Diversity Pool")
plt.plot(ecpf_ht["classifications correct (percent)"], label="ECPF with Hoeffding Tree")
plt.plot(ecpf_arf["classifications correct (percent)"], label="ECPF with ARF")
plt.plot(pearl["classifications correct (percent)"], label="PEARL", linestyle="--")

plt.legend()
plt.xlabel("no. of instances")
plt.ylabel("accuracy")

plt.show()
# plt.savefig('covtype-results.png', bbox_inches='tight', dpi=100)
