#!/usr/bin/env python3

import copy
import sys
import math
import argparse
import numpy as np
from collections import defaultdict, deque

from stream_generators import *
from LRU_state import *

from sklearn.metrics import cohen_kappa_score

from arf_hoeffding_tree import ARFHoeffdingTree
from skmultiflow.drift_detection.adwin import ADWIN

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["backend"] = "Qt4Agg"
plt.rcParams["figure.figsize"] = (20, 10)

class AdaptiveTree(object):
    def __init__(self,
                 tree_pool_id,
                 foreground_idx,
                 arf_max_features,
                 warning_delta,
                 drift_delta):
        self.tree_pool_id = tree_pool_id
        self.foreground_idx = foreground_idx
        self.fg_tree = ARFHoeffdingTree(max_features=arf_max_features)
        self.bg_tree = None
        self.is_candidate = False
        self.warning_detector = ADWIN(args.warning_delta)
        self.drift_detector = ADWIN(args.drift_delta)
        self.predicted_labels= deque(maxlen=args.kappa_window)
        self.kappa = -sys.maxsize

    def reset():
        self.foreground_idx = -1
        self.is_candidate = False
        self.kappa = -sys.maxsize
        self.predicted_labels = deque(maxlen=args.kappa_window)

def predict(X, y, trees, should_vote):
    predictions = []

    for row in X:
        votes = defaultdict(int)
        for tree in trees:
            prediction = tree.fg_tree.predict([row])[0]
            tree.predicted_labels.append(prediction) # for kappa calculation

            if prediction == y[0]:
                tree.warning_detector.add_element(0)
                tree.drift_detector.add_element(0)
            else:
                tree.warning_detector.add_element(1)
                tree.drift_detector.add_element(1)

            votes[prediction] += 1

        if should_vote:
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

def prequantial_evaluation(stream, adaptive_trees, lru_states, cur_state):
    correct = 0
    x_axis = []
    accuracy_list = []
    actual_labels = deque(maxlen=args.kappa_window) # a window of size arg.kappa_window

    current_state = []
    candidate_trees = []

    tree_pool = [None] * args.tree_pool_size
    next_tree_id = args.tree_pool_size

    with open('hyperplane.csv', 'w') as data_out, open('results.csv', 'w') as out:
        # pretrain
        X, y = stream.next_sample(args.wait_samples * 3)
        partial_fit(X, y, adaptive_trees)

        for row in X:
            features = ",".join(str(v) for v in row)
            data_out.write(f"{features},{str(y[0])}\n")

        for count in range(0, args.max_samples):
            X, y = stream.next_sample()
            actual_labels.append(y[0])

            # test
            prediction = predict(X, y, adaptive_trees, should_vote=True)[0]

            # test on candidate trees
            predict(X, y, candidate_trees, should_vote=False)

            if prediction == y[0]:
                correct += 1

            target_state = copy.deepcopy(cur_state)
            drifted_tree_list = []

            for tree in adaptive_trees:

                if tree.warning_detector.detected_change():
                    tree.warning_detector.reset()
                    tree.bg_tree = ARFHoeffdingTree(arf_max_features)
                    target_state[tree.tree_pool_id] = '2'

                if tree.drift_detector.detected_change():
                    tree.drift_detector.reset()
                    drifted_tree_list.append(tree)

                    if tree.bg_tree is None:
                        tree.fg_tree = ARFHoeffdingTree(arf_max_features)
                    else:
                        tree.fg_tree = tree.bg_tree
                        tree.bg_tree = None

            closest_state = lru_states.get_closest_state(target_state)

            if len(closest_state) != 0:
                for i in range(0, repo_size):

                    if cur_state[i] == '0' \
                            and closest_state[i] == '1' \
                            and not tree_pool[i].is_candidate:

                        if len(candidate_trees) >= args.num_trees:
                            worst_candidate = candidate_trees.pop(0)
                            worst_candidate.reset()

                        tree_pool[i].is_candidate = True
                        candidate_trees.append(tree_pool[i])

            if len(drifted_tree_list) > 0:
                # sort candidates by kappa
                for candidate_tree in candidate_trees:
                    if candidate_tree.predicted_labels < args.kappa_window:
                        candidate_tree.kappa = -sys.maxsize
                    else:
                        candidate_tree.kappa = cohen_kappa_score(actual_labels,
                                                                 candidate_tree.predicted_labels)
                candidate_trees.sort(key=lambda c : c.kappa)

                next_state = cur_state

            if (count % args.wait_samples == 0) and (count != 0):
                accuracy = correct / args.wait_samples
                print(accuracy)

                x_axis.append(count)
                accuracy_list.append(accuracy)
                out.write(f"{count},{accuracy}")
                correct = 0

            # train
            partial_fit(X, y, adaptive_trees)

            features = ",".join(str(v) for v in X[0])
            data_out.write(f"{features},{str(y[0])}\n")

    return x_axis, accuracy_list

def evaluate():
    fig, ax = plt.subplots(2, 2, sharey=True, constrained_layout=True)

    stream = prepare_hyperplane_streams(noise_1=0.05, noise_2=0.1)
    stream.prepare_for_use()
    print(stream.get_data_info())

    adaptive_trees = [AdaptiveTree(tree_pool_id=i,
                                   foreground_idx=i,
                                   arf_max_features=arf_max_features,
                                   warning_delta=args.warning_delta,
                                   drift_delta=args.drift_delta) for i in range(0, args.num_trees)]

    cur_state = ['1' if i < args.num_trees else '0' for i in range(0, repo_size)]

    lru_states = LRU_state(capacity=repo_size, distance_threshold=100)
    lru_states.enqueue(cur_state)

    x_axis, accuracy_list = prequantial_evaluation(stream,
                                                   adaptive_trees,
                                                   lru_states,
                                                   cur_state)

    ax[0, 0].plot(x_axis, accuracy_list)
    ax[0, 0].set_title("Accuracy")
    plt.xlabel("no. of instances")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tree",
                        dest="num_trees", default=1, type=int,
                        help="number of trees in the forest")
    parser.add_argument("-p", "--pool",
                        dest="tree_pool_size", default=1, type=int,
                        help="number of trees in the online tree repository")
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
    parser.add_argument("--kappa_window",
                        dest="kappa_window", default=50, type=int,
                        help="number of instances must be seen for calculating kappa")
    args = parser.parse_args()

    print(f"num_trees: {args.num_trees}")
    print(f"warning_delta: {args.warning_delta}")
    print(f"drift_delta: {args.drift_delta}")
    print(f"max_samples: {args.max_samples}")
    print(f"wait_samples: {args.wait_samples}")
    print(f"kappa_window: {args.kappa_window}")

    num_classes = 2
    arf_max_features = int(math.log2(num_classes)) + 1
    repo_size = args.num_trees * 4

    evaluate()
