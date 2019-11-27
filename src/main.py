#!/usr/bin/env python3

import copy
import sys
import math
import argparse
from collections import defaultdict, deque
import random
import pathlib
import time
import os.path
import logging

import numpy as np
from sklearn.metrics import cohen_kappa_score
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.trees.arf_hoeffding_tree import ARFHoeffdingTree
from skmultiflow.data.file_stream import FileStream

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["backend"] = "Qt4Agg"
plt.rcParams["figure.figsize"] = (20, 10)

from stream_generator import *
from LRU_state import *
from state_graph import *

class AdaptiveTree(object):
    def __init__(self,
                 tree,
                 tree_pool_id=-1):
        self.tree_pool_id = tree_pool_id
        self.tree = tree
        self.bg_adaptive_tree = None
        self.is_candidate = False
        self.warning_detector = ADWIN(args.warning_delta)
        self.drift_detector = ADWIN(args.drift_delta)
        self.predicted_labels = deque(maxlen=args.kappa_window)
        self.kappa = -sys.maxsize

    def update_kappa(self, actual_labels):
        if len(self.predicted_labels) < args.kappa_window:
            self.kappa = -sys.maxsize
        else:
            self.kappa = cohen_kappa_score(actual_labels, self.predicted_labels)
        return self.kappa

    def reset(self):
        self.bg_adaptive_tree = None
        self.is_candidate = False
        self.warning_detector.reset()
        self.drift_detector.reset()
        self.predicted_labels.clear()
        self.kappa = -sys.maxsize


class GraphSwitch:
    def __init__(self, window_size, state_graph, reuse_rate):
        self.window_size = window_size
        self.candidate_tree_count = 0
        self.total_tree_count = 0
        self.state_graph = state_graph
        self.reuse_rate = reuse_rate
        if window_size > 0:
            self.window = deque(maxlen=window_size)

    def update(self, value):
        self.candidate_tree_count += value
        self.total_tree_count += 1

        if self.window_size <= 0:
            return

        if len(self.window) >= self.window_size:
            self.candidate_tree_count -= self.window[0]
        self.window.append(value)

    def switch(self):
        cur_reuse_rate = 0
        if self.window_size <= 0:
            cur_reuse_rate = self.candidate_tree_count / self.total_tree_count
        else:
            cur_reuse_rate = self.candidate_tree_count / self.window_size

        if cur_reuse_rate >= self.reuse_rate:
            self.state_graph.is_stable = True
        else:
            self.state_graph.is_stable = False


def update_drift_detector(adaptive_tree, predicted_label, actual_label):
    if predicted_label == actual_label:
        adaptive_tree.warning_detector.add_element(0)
        adaptive_tree.drift_detector.add_element(0)
    else:
        adaptive_tree.warning_detector.add_element(1)
        adaptive_tree.drift_detector.add_element(1)

def predict(X, y, adaptive_trees, should_vote):
    predictions = []

    for i in range(0, len(X)):
        feature_row = X[i]
        label = y[i]

        votes = defaultdict(int)
        for adaptive_tree in adaptive_trees:
            predicted_label = adaptive_tree.tree.predict([feature_row])[0]

            adaptive_tree.predicted_labels.append(predicted_label) # for kappa calculation
            if should_vote:
                update_drift_detector(adaptive_tree, predicted_label, label)

            votes[predicted_label] += 1

            # background tree needs to predict for performance measurement
            if adaptive_tree.bg_adaptive_tree is not None:
                predict([feature_row], [label], [adaptive_tree.bg_adaptive_tree], False)

        if should_vote:
            predictions.append(max(votes, key=votes.get))

    return predictions

def partial_fit(X, y, adaptive_trees):
    for i in range(0, len(X)):
        for adaptive_tree in adaptive_trees:
            n = np.random.poisson(1)
            adaptive_tree.tree.partial_fit([X[i]], [y[i]], sample_weight=[n])
            if adaptive_tree.bg_adaptive_tree is not None:
                adaptive_tree.bg_adaptive_tree.tree.partial_fit([X[i]], [y[i]], sample_weight=[n])

def update_candidate_trees(candidate_trees,
                           tree_pool,
                           cur_state,
                           closest_state,
                           cur_tree_pool_size):
    if len(closest_state) == 0:
        return

    # print(f"closest_state {closest_state}")
    # print(f"cur_state {cur_state}")

    for i in range(0, cur_tree_pool_size):

        if cur_state[i] == '0' \
                and closest_state[i] == '1' \
                and not tree_pool[i].is_candidate:

            if len(candidate_trees) >= args.num_trees:
                worst_candidate = candidate_trees.pop(0)
                worst_candidate.reset()

            tree_pool[i].is_candidate = True
            candidate_trees.append(tree_pool[i])

    # print("candidate_trees", [c.tree_pool_id for c in candidate_trees])

def select_candidate_trees(count,
                           target_state,
                           warning_tree_id_list,
                           candidate_trees,
                           tree_pool,
                           lru_states,
                           state_graph,
                           cur_state,
                           cur_tree_pool_size):

    if args.enable_state_graph:
        # try trigger lossy counting
        if state_graph.update(len(warning_tree_id_list)):
            logger.info(f"{count},lossy counting triggered")

    if state_graph.is_stable:
        for warning_tree_id in warning_tree_id_list:
            # print("finding next_id...")
            next_id = state_graph.get_next_tree_id(warning_tree_id)
            if next_id == -1:
                # print(f"tree {warning_tree_id} does not have next id")
                state_graph.is_stable = False

            else:
                # print("Next tree found, adding candidate tree...")
                if not tree_pool[next_id].is_candidate:
                    candidate_trees.append(tree_pool[next_id])
                    # print("candidate tree added")

    if not state_graph.is_stable:
        logger.info(f"{count},pattern matching")

        # trigger pattern matching
        closest_state = lru_states.get_closest_state(target_state)

        update_candidate_trees(candidate_trees=candidate_trees,
                               tree_pool=tree_pool,
                               cur_state=cur_state,
                               closest_state=closest_state,
                               cur_tree_pool_size=cur_tree_pool_size)
    else:
        logger.info(f"{count},graph transition")

def update_reuse_rate(background_count, candidate_count, state_graph):
    global background_reuse_total_count
    global candidate_reuse_total_count
    global background_reuse_window
    global candidate_reuse_window

    if len(background_reuse_window) >= args.reuse_window_size:
        background_reuse_total_count -= background_reuse_window[0]
        candidate_reuse_total_count -= candidate_reuse_window[0]

    background_reuse_total_count += background_count
    candidate_reuse_total_count += candidate_count

    background_reuse_window.append(background_count)
    candidate_reuse_window.append(candidate_count)

    total_reuse_count = background_reuse_total_count + candidate_reuse_total_count
    reuse_rate = 0.0
    if total_reuse_count != 0:
        reuse_rate = candidate_reuse_total_count / total_reuse_count

    with open(f"{result_directory}/reuse-rate-{args.generator_seed}.log", 'a') as out:
        out.write(f"{background_reuse_total_count},{candidate_reuse_total_count},{reuse_rate}\n")
        out.flush()

    if args.enable_state_graph:
        if reuse_rate >= args.reuse_rate_upper_bound:
            state_graph.is_stable = True

        if reuse_rate < args.reuse_rate_lower_bound:
            state_graph.is_stable = False

def adapt_state(drifted_tree_list,
                candidate_trees,
                tree_pool,
                state_graph,
                graph_switch,
                cur_state,
                cur_tree_pool_size,
                adaptive_trees,
                drifted_tree_pos,
                actual_labels):

    # print("Drifts detected. Adapting states for", [t.tree_pool_id for t in drifted_tree_list])

    # sort candidates by kappa
    for candidate_tree in candidate_trees:
        candidate_tree.update_kappa(actual_labels)
    candidate_trees.sort(key=lambda c : c.kappa)

    for drifted_tree in drifted_tree_list:
        # TODO
        if cur_tree_pool_size >= repo_size:
            print("early break")
            exit()

        drifted_tree.update_kappa(actual_labels)
        swap_tree = drifted_tree

        background_count = 0
        candidate_count = 0

        if len(candidate_trees) > 0 \
                and candidate_trees[-1].kappa - drifted_tree.kappa >= args.cd_kappa_threshold:
            # swap drifted tree with the candidate tree
            swap_tree = candidate_trees.pop()
            swap_tree.is_candidate = False

            # candidate_count += 1
            if args.enable_state_graph:
                graph_switch.update(1)

        if swap_tree is drifted_tree:
            add_to_repo = True
            # background_count += 1
            if args.enable_state_graph:
                graph_switch.update(0)

            if drifted_tree.bg_adaptive_tree is None:
                    swap_tree = \
                        AdaptiveTree(tree=ARFHoeffdingTree(max_features=arf_max_features))

            else:
                prediction_win_size = len(drifted_tree.bg_adaptive_tree.predicted_labels)
                # print(f"bg_tree window size: {prediction_win_size}")

                drifted_tree.bg_adaptive_tree.update_kappa(actual_labels)
                # print(f"bg_tree kappa: {drifted_tree.bg_adaptive_tree.kappa} "
                #       f"swap_tree.kappa: {swap_tree.kappa}")

                if drifted_tree.bg_adaptive_tree.kappa == -sys.maxsize:
                    # add bg_adaptive_tree to the repo even if it didn't fill the window
                    swap_tree = drifted_tree.bg_adaptive_tree

                elif drifted_tree.bg_adaptive_tree.kappa - swap_tree.kappa >= args.bg_kappa_threshold:
                    swap_tree = drifted_tree.bg_adaptive_tree

                else:
                    # false positive
                    add_to_repo = False

            if add_to_repo:
                swap_tree.reset()

                # assign a new tree_pool_id for background tree
                # and add background tree to tree_pool
                swap_tree.tree_pool_id = cur_tree_pool_size
                tree_pool[cur_tree_pool_size] = swap_tree
                cur_tree_pool_size += 1

        cur_state[drifted_tree.tree_pool_id] = '0'
        cur_state[swap_tree.tree_pool_id] = '1'

        if args.enable_state_graph:
            state_graph.add_edge(drifted_tree.tree_pool_id, swap_tree.tree_pool_id)

        # replace drifted tree with swap tree
        pos = drifted_tree_pos.pop()
        adaptive_trees[pos] = swap_tree
        drifted_tree.reset()

    if args.enable_state_graph:
        graph_switch.switch()
    # if args.enable_state_adaption:
        # update_reuse_rate(background_count, candidate_count, state_graph)

    return cur_tree_pool_size

def prequential_evaluation(adaptive_trees,
                           lru_states,
                           state_graph,
                           graph_switch,
                           cur_state,
                           tree_pool):
    correct = 0
    x_axis = []
    accuracy_list = []
    actual_labels = deque(maxlen=args.kappa_window) # a window of size arg.kappa_window

    sample_counter = 0
    sample_counter_interval = 0
    window_accuracy = 0.0
    window_kappa = 0.0
    window_actual_labels = []
    window_predicted_labels = []

    current_state = []
    candidate_trees = []

    cur_tree_pool_size = args.num_trees

    with open(metric_output_file, 'w') as out:
        out.write("count,accuracy,kappa,memory\n")

        for count in range(0, args.max_samples):
            X, y = stream.next_sample()
            actual_labels.append(y[0])

            # test
            prediction = predict(X, y, adaptive_trees, should_vote=True)[0]

            # test on candidate trees
            predict(X, y, candidate_trees, should_vote=False)

            window_actual_labels.append(y[0])
            window_predicted_labels.append(prediction)
            if prediction == y[0]:
                correct += 1

            target_state = copy.deepcopy(cur_state)

            warning_tree_id_list = []
            drifted_tree_list = []
            drifted_tree_pos = []

            for i in range(0, args.num_trees):

                tree = adaptive_trees[i]
                warning_detected_only = False
                if tree.warning_detector.detected_change():
                    warning_detected_only = True
                    tree.warning_detector.reset()

                    tree.bg_adaptive_tree = \
                        AdaptiveTree(tree=ARFHoeffdingTree(max_features=arf_max_features))

                if tree.drift_detector.detected_change():
                    warning_detected_only = False
                    tree.drift_detector.reset()
                    drifted_tree_list.append(tree)
                    drifted_tree_pos.append(i)

                    if not args.enable_state_adaption:
                        if tree.bg_adaptive_tree is None:
                            tree.tree = ARFHoeffdingTree(max_features=arf_max_features)
                        else:
                            tree.tree = tree.bg_adaptive_tree.tree
                        tree.reset()

                if warning_detected_only:
                    warning_tree_id_list.append(tree.tree_pool_id)
                    target_state[tree.tree_pool_id] = '2'

            if args.enable_state_adaption:
                # if warnings are detected, find closest state and update candidate_trees list
                if len(warning_tree_id_list) > 0:
                    select_candidate_trees(count=count,
                                           target_state=target_state,
                                           warning_tree_id_list=warning_tree_id_list,
                                           candidate_trees=candidate_trees,
                                           tree_pool=tree_pool,
                                           lru_states=lru_states,
                                           state_graph=state_graph,
                                           cur_state=cur_state,
                                           cur_tree_pool_size=cur_tree_pool_size)

                # if actual drifts are detected, swap trees and update cur_state
                if len(drifted_tree_list) > 0:
                    cur_tree_pool_size = adapt_state(drifted_tree_list=drifted_tree_list,
                                                     candidate_trees=candidate_trees,
                                                     tree_pool=tree_pool,
                                                     state_graph=state_graph,
                                                     graph_switch=graph_switch,
                                                     cur_state=cur_state,
                                                     cur_tree_pool_size=cur_tree_pool_size,
                                                     adaptive_trees=adaptive_trees,
                                                     drifted_tree_pos=drifted_tree_pos,
                                                     actual_labels=actual_labels)

                lru_states.enqueue(cur_state)
                # print(f"Add state: {cur_state}")

            if (count % args.wait_samples == 0) and (count != 0):
                accuracy = correct / args.wait_samples

                window_accuracy = (window_accuracy * sample_counter + accuracy) \
                    / (sample_counter + 1)

                kappa = cohen_kappa_score(window_actual_labels, window_predicted_labels)
                window_kappa = (window_kappa * sample_counter + kappa) \
                        / (sample_counter + 1)

                sample_counter += 1
                sample_counter_interval += args.wait_samples
                correct = 0

                if sample_counter_interval == args.sample_freq:
                    x_axis.append(count)
                    accuracy_list.append(window_accuracy)

                    memory_usage = 0
                    if args.enable_state_adaption:
                        memory_usage = lru_states.get_size()
                    if args.enable_state_graph:
                        memory_usage += state_graph.get_size()
                    print(f"{count},{window_accuracy},{window_kappa},{memory_usage}")
                    out.write(f"{count},{window_accuracy},{window_kappa},{memory_usage}\n")
                    out.flush()

                    sample_counter = 0
                    sample_counter_interval = 0

                    window_accuracy = 0.0
                    window_kappa = 0.0
                    window_actual_labels = []
                    window_predicted_labels = []

            # train
            partial_fit(X, y, adaptive_trees)

    print(f"length of candidate_trees: {len(candidate_trees)}")
    return x_axis, accuracy_list

def evaluate():
    # fig, ax = plt.subplots(2, 2, sharey=True, constrained_layout=True)

    adaptive_trees = [AdaptiveTree(tree_pool_id=i,
                                   tree=ARFHoeffdingTree(max_features=arf_max_features)
                      ) for i in range(0, args.num_trees)]

    cur_state = ['1' if i < args.num_trees else '0' for i in range(0, repo_size)]

    lru_states = LRU_state(capacity=repo_size, edit_distance_threshold=args.edit_distance_threshold)
    lru_states.enqueue(cur_state)

    state_graph = LossyStateGraph(repo_size, args.lossy_window_size)

    graph_switch = GraphSwitch(window_size=args.reuse_window_size,
                               state_graph=state_graph,
                               reuse_rate=args.reuse_rate_upper_bound)

    tree_pool = [None] * repo_size
    for i in range(0, args.num_trees):
        tree_pool[i] = adaptive_trees[i]

    x_axis, accuracy_list = prequential_evaluation(adaptive_trees,
                                                   lru_states,
                                                   state_graph,
                                                   graph_switch,
                                                   cur_state,
                                                   tree_pool)

    # ax[0, 0].plot(x_axis, accuracy_list)
    # ax[0, 0].set_title("Accuracy")
    # plt.xlabel("no. of instances")
    # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tree",
                        dest="num_trees", default=60, type=int,
                        help="number of trees in the forest")
    parser.add_argument("-g", "--generator",
                        dest="generator", default="agrawal", type=str,
                        help="name of the synthetic data generator")
    # parser.add_argument("-o", "--output",
    #                     dest="metric_output_file", default="result", type=str,
    #                     help="output path for metrics")
    # parser.add_argument("--pool",
    #                     dest="tree_pool_size", default=180, type=int,
    #                     help="number of trees in the online tree repository")
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
    parser.add_argument("--kappa_window",
                        dest="kappa_window", default=50, type=int,
                        help="number of instances must be seen for calculating kappa")
    parser.add_argument("--random_state",
                        dest="random_state", default=0, type=int,
                        help="Seed used for adaptive hoeffding tree")
    parser.add_argument("--generator_seed",
                        dest="generator_seed", default=0, type=int,
                        help="Seed used for generating synthetic data")
    parser.add_argument("--enable_generator_noise",
                        dest="enable_generator_noise", action="store_true",
                        help="Enable noise in synthetic data generator")
    parser.set_defaults(enable_generator_noise=False)

    parser.add_argument("-s", "--enable_state_adaption",
                        dest="enable_state_adaption", action="store_true",
                        help="enable the state adaption algorithm")
    parser.set_defaults(enable_state_adaption=False)
    parser.add_argument("-p", "--enable_state_graph",
                        dest="enable_state_graph", action="store_true",
                        help="enable state transition graph")
    parser.set_defaults(enable_state_graph=False)

    parser.add_argument("--cd_kappa_threshold",
                        dest="cd_kappa_threshold", default=0.2, type=float,
                        help="Kappa value that the candidate tree needs to outperform both"
                             "background tree and foreground drifted tree")
    parser.add_argument("--bg_kappa_threshold",
                        dest="bg_kappa_threshold", default=0.00, type=float,
                        help="Kappa value that the background tree needs to outperform the "
                             "foreground drifted tree to prevent from false positive")
    parser.add_argument("--edit_distance_threshold",
                        dest="edit_distance_threshold", default=100, type=int,
                        help="The maximum edit distance threshold")
    parser.add_argument("--lossy_window_size",
                        dest="lossy_window_size", default=5, type=int,
                        help="Window size for lossy count")
    parser.add_argument("--reuse_window_size",
                        dest="reuse_window_size", default=0, type=int,
                        help="Window size for calculating reuse rate")
    parser.add_argument("--reuse_rate_upper_bound",
                        dest="reuse_rate_upper_bound", default=0.4, type=float,
                        help="The reuse rate threshold for switching from "
                             "pattern matching to graph transition")
    parser.add_argument("--reuse_rate_lower_bound",
                        dest="reuse_rate_lower_bound", default=0.1, type=float,
                        help="The reuse rate threshold for switching from "
                             "pattern matching to graph transition")

    args = parser.parse_args()

    if args.reuse_rate_upper_bound < args.reuse_rate_lower_bound:
        exit("reuse rate upper bound must be greater than or equal to the lower bound")

    if args.enable_state_graph:
        args.enable_state_adaption = True

    stream = None
    potential_file = f"../data/{args.generator}/{args.generator}-{args.generator_seed}.csv"

    # prepare data
    if os.path.isfile(potential_file):
        print(f"preparing stream from file {potential_file}...")
        stream = FileStream(potential_file)
        stream.prepare_for_use()

        # args.max_samples = stream.n_remaining_samples()
        args.max_samples = min(args.max_samples, stream.n_remaining_samples())

    else:
        print(f"preparing stream from {args.generator} generator...")
        # concepts = [v for v in range(0, 10)]
        concepts = [4,0,8]
        stream = RecurrentDriftStream(generator=args.generator,
                                      concepts=concepts,
                                      has_noise=args.enable_generator_noise,
                                      random_state=args.generator_seed)
        stream.prepare_for_use()
        print(stream.get_data_info())

    result_directory = args.generator
    if args.enable_generator_noise:
        result_directory = f"{result_directory}-noise"

    metric_output_file = "result"
    time_output_file = "time"

    if args.enable_state_graph:
        result_directory = f"{result_directory}/" \
                           f"k{args.cd_kappa_threshold}-e{args.edit_distance_threshold}/" \
                           f"r{args.reuse_rate_upper_bound}-r{args.reuse_rate_lower_bound}-" \
                           f"w{args.reuse_window_size}/" \
                           f"lossy-{args.lossy_window_size}"

        metric_output_file = f"{metric_output_file}-parf"
        time_output_file = f"{time_output_file}-parf"

    elif args.enable_state_adaption:
        result_directory = f"{result_directory}/" \
                           f"k{args.cd_kappa_threshold}-e{args.edit_distance_threshold}/"

        metric_output_file = f"{metric_output_file}-sarf"
        time_output_file = f"{time_output_file}-sarf"

    pathlib.Path(result_directory).mkdir(parents=True, exist_ok=True)

    metric_output_file = f"{result_directory}/{metric_output_file}-{args.generator_seed}.csv"
    time_output_file = f"{result_directory}/{time_output_file}-{args.generator_seed}.log"


    configs = (
        f"metric_output_file: {metric_output_file}\n"
        f"warning_delta: {args.warning_delta}\n"
        f"drift_delta: {args.drift_delta}\n"
        f"max_samples: {args.max_samples}\n"
        f"wait_samples: {args.wait_samples}\n"
        f"sample_freq: {args.sample_freq}\n"
        f"kappa_window: {args.kappa_window}\n"
        f"random_state: {args.random_state}\n"
        f"enable_state_adaption: {args.enable_state_adaption}\n"
        f"enable_state_graph: {args.enable_state_graph}\n")

    print(configs)
    with open(f"{result_directory}/config", 'w') as out:
        out.write(configs)
        out.flush()

    num_features = stream.n_features
    arf_max_features = int(math.log2(num_features)) + 1

    repo_size = args.num_trees * 160
    np.random.seed(args.random_state)
    random.seed(0)

    background_reuse_total_count = 0
    candidate_reuse_total_count = 0
    # background_reuse_window = deque(maxlen=args.reuse_window_size)
    # candidate_reuse_window = deque(maxlen=args.reuse_window_size)

    if args.enable_state_adaption:
        with open(f"{result_directory}/reuse-rate-{args.generator_seed}.log", 'w') as out:
            out.write("background_window_count,candidate_window_count,reuse_rate\n")

    logging.basicConfig(filename=f'{result_directory}/processes-{args.generator_seed}.info',
                        format='%(message)s',
                        filemode='w')

    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)

    start = time.process_time()
    evaluate()
    elapsed = time.process_time() - start

    with open(f"{time_output_file}", 'w') as out:
        out.write(str(elapsed) + '\n')
