#!/usr/bin/env python3

import sys
import math
import argparse
import numpy as np
from collections import defaultdict

from stream_generators import *
from LRU_state import *

from arf_hoeffding_tree import ARFHoeffdingTree
from skmultiflow.drift_detection.adwin import ADWIN

from scipy.stats import ks_2samp

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["backend"] = "Qt4Agg"
plt.rcParams["figure.figsize"] = (20, 10)

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
            n = np.random.poisson(1)
            for j in range(0, n):
                tree.partial_fit([X[i]], [y[i]])

def prequantial_evaluation(stream,
                           fg_trees,
                           bg_trees,
                           warning_detectors,
                           drift_detectors,
                           states):
    correct = 0
    x_axis = []
    accuracy_list = []

    # pretrain
    X, y = stream.next_sample(args.wait_samples * 3)
    partial_fit(X, y, fg_trees)

    for count in range(0, args.max_samples):
        X, y = stream.next_sample()

        # test
        prediction = predict(X, fg_trees)[0]

        if prediction == y[0]:
            correct += 1

        if (count % args.wait_samples == 0) and (count != 0):
            accuracy = correct / args.wait_samples
            print(accuracy)

            x_axis.append(count)
            accuracy_list.append(accuracy)
            correct = 0

        # train
        partial_fit(X, y, fg_trees)

    return x_axis, accuracy_list

def evaluate():
    fig, ax = plt.subplots(2, 2, sharey=True, constrained_layout=True)

    stream = prepare_hyperplane_streams(noise_1=0.05, noise_2=0.1)
    stream.prepare_for_use()
    print(stream.get_data_info())

    num_classes = 2
    arf_max_features = int(math.log2(num_classes)) + 1

    repo_size = args.num_trees * 4
    states = LRU_state(repo_size)

    fg_trees = []
    for i in range(0, args.num_trees):
        fg_trees.append(ARFHoeffdingTree(max_features=arf_max_features))
    bg_trees = [None] * args.num_trees

    warning_detectors = [ADWIN(args.warning_delta)] * args.num_trees
    drift_detectors = [ADWIN(args.drift_delta)] * args.num_trees

    x_axis, accuracy_list = prequantial_evaluation(stream,
                                                   fg_trees,
                                                   bg_trees,
                                                   warning_detectors,
                                                   drift_detectors,
                                                   states)

    ax[0, 0].plot(x_axis, accuracy_list)
    ax[0, 0].set_title("Accuracy")
    plt.xlabel("no. of instances")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tree",
                        dest="num_trees", default=1, type=int,
                        help="number of trees in the forest")
    parser.add_argument("-w", "--warning",
                        dest="warning_delta", default=0.001, type=float,
                        help="delta value for drift warning detector")
    parser.add_argument("-d", "--drift",
                        dest="drift_delta", default=0.0001, type=float,
                        help="delta value for drift detector")
    parser.add_argument("--max_samples",
                        dest="max_samples", default=10000, type=int,
                        help="total number of samples")
    parser.add_argument("--wait_samples",
                        dest="wait_samples", default=100, type=int,
                        help="number of samples per evaluation")
    args = parser.parse_args()

    print(f"num_trees: {args.num_trees}")
    print(f"warning_delta: {args.warning_delta}")
    print(f"drift_delta: {args.drift_delta}")
    print(f"max_samples: {args.max_samples}")
    print(f"wait_samples: {args.wait_samples}")

    evaluate()
