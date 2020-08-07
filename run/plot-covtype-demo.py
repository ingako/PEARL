#!/usr/bin/env python

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["backend"] = "Qt4Agg"
plt.rcParams["figure.figsize"] = (20, 10)

dataset="covtype"
kappa=0.4
ed=90
reuse_window=0
reuse_rate=0.18
lossy_window=100000000

arf_cpp_path = f"{dataset}/result-0.csv"
pearl_cpp_path = f"{dataset}/k{kappa}-e{ed}/result-sarf-0.csv"
# pearl_cpp_path = f"{dataset}/k{kappa}-e{ed}/r{reuse_rate}-r{reuse_rate}-w{reuse_window}/lossy-{lossy_window}/result-parf-0.csv"

arf_cpp = pd.read_csv(arf_cpp_path, index_col=0)
pearl_cpp = pd.read_csv(pearl_cpp_path, index_col=0)

gain = sum(pearl_cpp["accuracy"]) - sum(arf_cpp["accuracy"])
print(f"PEARL's cumulative accuracy gain: {gain}")

plt.plot(arf_cpp["accuracy"], label="ARF")
plt.plot(pearl_cpp["accuracy"], label="PEARL", linestyle="--")

plt.legend()
plt.xlabel("no. of instances")
plt.ylabel("accuracy")

plt.show()
# plt.savefig('covtype-results.png', bbox_inches='tight', dpi=100)
