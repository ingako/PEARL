#!/usr/bin/env python3

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["backend"] = "Qt4Agg"
plt.rcParams["figure.figsize"] = (20, 10)

sarf_results_path = "results.csv"
arf_results_path = "results_arf.csv"

sarf = pd.read_csv(sarf_results_path, header=None, index_col=0)
arf = pd.read_csv(arf_results_path, header=None, index_col=0)

plt.plot(sarf, label="sarf")
plt.plot(arf, label="arf")
plt.legend()

plt.title("Accuracy")
plt.xlabel("no. of instances")
plt.ylabel("accuracy")

plt.ylim(0, 1)
# plt.xlim(0, 10)

plt.show()
