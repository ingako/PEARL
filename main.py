#!/usr/bin/env python3

import sys
import math
import numpy as np
from collections import defaultdict
from stream_generators import *

from skmultiflow.trees import HoeffdingTree
from skmultiflow.drift_detection.adwin import ADWIN

from scipy.stats import ks_2samp

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["backend"] = "Qt4Agg"
plt.rcParams["figure.figsize"] = (20, 10)

fig, ax = plt.subplots(2, 2, sharey=True, constrained_layout=True)

def predict(X, trees):
    predictions = []

    for row in X:
        votes = defaultdict(int)
        for tree in trees:
            prediction = tree.predict([row])[0]
            votes[prediction] += 1
        predictions.append(max(votes, key=votes.get))

    return predictions

def partial_fit(X, y, trees):
    for i in range(0, len(X)):
        for tree in trees:
            if tree == None:
                continue
            tree.partial_fit([X[i]], [y[i]])

stream = prepare_hyperplane_streams(noise_1=0.05, noise_2=0.1)
stream.prepare_for_use()
print(stream.get_data_info())

num_classes = 2
num_trees = int(sys.argv[1])
warning_delta = 0.001
drift_delta = 0.0001

fg_trees = [HoeffdingTree()]
bg_trees = [None] * num_trees
warning_detectors = [ADWIN(warning_delta)] * num_trees
drift_detectors = [ADWIN(drift_delta)] * num_trees

max_samples = 10000
wait_samples = 100

correct = 0
x_axis = []
accuracy_list = []

# pretrain
X, y = stream.next_sample(wait_samples * 3)
partial_fit(X, y, fg_trees)

for count in range(0, max_samples):
    X, y = stream.next_sample()

    # test
    prediction = predict(X, fg_trees)[0]

    if prediction == y[0]:
        correct += 1

    if (count % wait_samples == 0) and (count != 0):
        accuracy = correct / wait_samples
        print(accuracy)

        x_axis.append(count)
        accuracy_list.append(accuracy)
        correct = 0

    # train
    partial_fit(X, y, fg_trees)

ax[0, 0].plot(x_axis, accuracy_list)
ax[0, 0].set_title("Accuracy")
plt.xlabel("no. of instances")
plt.show()
