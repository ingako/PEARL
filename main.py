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

class adaptive_tree(object):
    def __init__(self, arf_max_features, warning_delta, drift_delta):
        self.fg_tree = ARFHoeffdingTree(max_features=arf_max_features)
        self.bg_tree = None
        self.warning_detector = ADWIN(args.warning_delta)
        self.drift_detector = ADWIN(args.drift_delta)

def predict(X, y, trees):
    predictions = []

    for row in X:
        votes = defaultdict(int)
        for tree in trees:
            prediction = tree.fg_tree.predict([row])[0]
            if prediction == y[0]:
                tree.warning_detector.add_element(0)
                tree.drift_detector.add_element(0)
            else:
                tree.warning_detector.add_element(1)
                tree.drift_detector.add_element(1)

            votes[prediction] += 1

        predictions.append(max(votes, key=votes.get))

    return predictions

def partial_fit(X, y, trees):
    for i in range(0, len(X)):
        for tree in trees:
            n = np.random.poisson(1)
            for j in range(0, n):
                tree.fg_tree.partial_fit([X[i]], [y[i]])
                if tree.bg_tree is not None:
                    tree.bg_tree.partial_fit([X[i]], [y[i]])


def prequantial_evaluation(stream,
                           adaptive_trees,
                           states):
    correct = 0
    x_axis = []
    accuracy_list = []

    with open('hyperplane.csv', 'w') as out:
        # pretrain
        X, y = stream.next_sample(args.wait_samples * 3)
        partial_fit(X, y, adaptive_trees)

        for row in X:
            features = ",".join(str(v) for v in row)
            out.write(f"{features},{str(y[0])}\n")

        for count in range(0, args.max_samples):
            X, y = stream.next_sample()

            # test
            prediction = predict(X, y, adaptive_trees)[0]

            if prediction == y[0]:
                correct += 1

            for tree in adaptive_trees:
                if tree.warning_detector.detected_change():
                    tree.warning_detector.reset()
                    tree.bg_tree = ARFHoeffdingTree(arf_max_features)
                if tree.drift_detector.detected_change():
                    tree.drift_detector.reset()
                    if tree.bg_tree is None:
                        tree.fg_tree = ARFHoeffdingTree(arf_max_features)
                    else:
                        tree.fg_tree = tree.bg_tree
                        tree.bg_tree = None

            if (count % args.wait_samples == 0) and (count != 0):
                accuracy = correct / args.wait_samples
                print(accuracy)

                x_axis.append(count)
                accuracy_list.append(accuracy)
                correct = 0

            # train
            partial_fit(X, y, adaptive_trees)

            features = ",".join(str(v) for v in X[0])
            out.write(f"{features},{str(y[0])}\n")

    return x_axis, accuracy_list

def evaluate():
    fig, ax = plt.subplots(2, 2, sharey=True, constrained_layout=True)

    stream = prepare_hyperplane_streams(noise_1=0.05, noise_2=0.1)
    stream.prepare_for_use()
    print(stream.get_data_info())

    states = LRU_state(repo_size)

    adaptive_trees = [adaptive_tree(arf_max_features,
                                    args.warning_delta,
                                    args.drift_delta) for _ in range(0, args.num_trees)]

    x_axis, accuracy_list = prequantial_evaluation(stream,
                                                   adaptive_trees,
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

    num_classes = 2
    arf_max_features = int(math.log2(num_classes)) + 1
    repo_size = args.num_trees * 4

    evaluate()
