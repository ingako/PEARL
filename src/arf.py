#!/usr/bin/env python3

import math
import argparse
import numpy as np

from stream_generators import *

from arf_hoeffding_tree import ARFHoeffdingTree
from skmultiflow.drift_detection.adwin import ADWIN

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["backend"] = "Qt4Agg"
plt.rcParams["figure.figsize"] = (20, 10)

class AdaptiveTree(object):
    def __init__(self, fg_tree):
        self.fg_tree = fg_tree
        self.bg_tree = None
        self.warning_detector = ADWIN(args.warning_delta)
        self.drift_detector = ADWIN(args.drift_delta)

def update_drift_detector(adaptive_tree, predicted_label, actual_label):
    if int(predicted_label) == int(actual_label):
        adaptive_tree.warning_detector.add_element(0)
        adaptive_tree.drift_detector.add_element(0)
    else:
        adaptive_tree.warning_detector.add_element(1)
        adaptive_tree.drift_detector.add_element(1)

def predict(X, y, trees):
    predictions = []

    for i in range(0, len(X)):
        feature_row = X[i]
        label = int(y[i])

        # votes = defaultdict(int)
        votes = {}
        for tree in trees:
            predicted_label = int(tree.fg_tree.predict(np.asarray([feature_row]))[0])
            try:
                votes[predicted_label] += 1
            except KeyError:
                votes[predicted_label] = 1

            update_drift_detector(tree, predicted_label, label)

        predictions.append(max(votes, key=votes.get))

    return predictions

def partial_fit(X, y, trees):
    for i in range(0, len(X)):
        for tree in trees:
            k = np.random.poisson(1)
            # for j in range(0, n):
            if k > 0:
                tree.fg_tree.partial_fit(np.asarray([X[i]]), np.asarray([y[i]]),
                                         sample_weight=np.asarray([k]))
                if tree.bg_tree is not None:
                    tree.bg_tree.partial_fit(np.asarray([X[i]]), np.asarray([y[i]]),
                                             sample_weight=np.asarray([k]))

def prequential_evaluation(stream, adaptive_trees):
    correct = 0
    x_axis = []
    accuracy_list = []

    sample_counter = 0
    window_accuracy = 0.0

    with open('results_arf.csv', 'w') as out:
        for count in range(0, 201): # args.max_samples):
            X, y = stream.next_sample()
            X = X.astype(int)
            y = y.astype(int)
            if count == 0:
                print(X)
                print(y)

            # test
            prediction = predict(X, y, adaptive_trees)[0]

            if int(prediction) == int(y[0]):
                correct += 1

            if (count % args.wait_samples == 0) and (count != 0):
                accuracy = correct / args.wait_samples
                # print(correct)
                # print(adaptive_trees[0].fg_tree.get_model_description())
                # print(adaptive_trees[0].fg_tree.get_model_measurements)
                correct = 0

                window_accuracy = (window_accuracy * sample_counter + accuracy) \
                    / (sample_counter + 1)
                sample_counter += args.wait_samples

                if sample_counter == args.sample_freq:
                    x_axis.append(count)
                    accuracy_list.append(window_accuracy)

                    print(f"{count},{window_accuracy}")
                    out.write(f"{count},{window_accuracy}\n")
                    out.flush()

                    sample_counter = 0
                    window_accuracy = 0.0

            warning_list = []
            drift_list = []
            for i in range(0, len(adaptive_trees)):
                tree = adaptive_trees[i]

                if tree.warning_detector.detected_change():
                    tree.bg_tree = tree.fg_tree.new_instance()
                    tree.warning_detector.reset()
                    warning_list.append(i)

                if tree.drift_detector.detected_change():
                    tree.drift_detector.reset()
                    print(f"{count}: replace bg tree")

                    if tree.bg_tree is None:
                        tree.fg_tree.reset()
                        tree.warning_detector.reset()
                    else:
                        tree.fg_tree = tree.bg_tree
                        tree.bg_tree = None

                    drift_list.append(i)

            # if len(warning_list) > 0:
            #     print(f"{count}-warning:{warning_list}")
            # if len(drift_list) > 0:
            #     print(f"{count}-drift:{drift_list}")

            # train
            partial_fit(X, y, adaptive_trees)

            # features = ",".join(str(v) for v in X[0])
            # data_out.write(f"{features},{str(y[0])}\n")

    return x_axis, accuracy_list

def evaluate():
    fig, ax = plt.subplots(2, 2, sharey=True, constrained_layout=True)

    # prepare data
    stream = prepare_data()
    print(stream.get_data_info())

    adaptive_trees = [AdaptiveTree(fg_tree=ARFHoeffdingTree(max_features=arf_max_features,
                                                            leaf_prediction='mc',
                                                            binary_split=True,
                                                            random_state=np.random)
                                  ) for i in range(0, args.num_trees)]

    x_axis, accuracy_list = prequential_evaluation(stream, adaptive_trees)

    # plot accuracy
    # ax[0, 0].plot(x_axis, accuracy_list)
    # ax[0, 0].set_title("Accuracy")
    # plt.xlabel("no. of instances")
    # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tree",
                        dest="num_trees", default=60, type=int,
                        help="number of trees in the forest")
    parser.add_argument("-w", "--warning",
                        dest="warning_delta", default=0.0001, type=float,
                        help="delta value for drift warning detector")
    parser.add_argument("-d", "--drift",
                        dest="drift_delta", default=0.00001, type=float,
                        help="delta value for drift detector")
    parser.add_argument("--max_samples",
                        dest="max_samples", default=200000, type=int,
                        help="total number of samples")
    parser.add_argument("--wait_samples",
                        dest="wait_samples", default=100, type=int,
                        help="number of samples per evaluation")
    parser.add_argument("--sample_freq",
                        dest="sample_freq", default=1000, type=int,
                        help="log interval for performance")
    parser.add_argument("--random_state",
                        dest="random_state", default=0, type=int,
                        help="Seed used for adaptive hoeffding tree")
    args = parser.parse_args()

    print(f"num_trees: {args.num_trees}")
    print(f"warning_delta: {args.warning_delta}")
    print(f"drift_delta: {args.drift_delta}")
    print(f"max_samples: {args.max_samples}")
    print(f"wait_samples: {args.wait_samples}")

    num_labels = 2
    num_features = 9
    arf_max_features = round(math.sqrt(num_features))
    print(f"max_features: {arf_max_features}")
    np.random.seed(args.random_state)

    evaluate()
