import copy
import sys
import math
from collections import defaultdict, deque

import numpy as np
from sklearn.metrics import cohen_kappa_score
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.trees.arf_hoeffding_tree import ARFHoeffdingTree

from stream_generator import *
from LRU_state import *
from state_graph import *

class AdaptiveTree(object):
    def __init__(self,
                 tree,
                 kappa_window,
                 warning_delta,
                 drift_delta,
                 tree_pool_id=-1):
        self.tree_pool_id = tree_pool_id
        self.tree = tree
        self.bg_adaptive_tree = None
        self.is_candidate = False
        self.warning_detector = ADWIN(warning_delta)
        self.drift_detector = ADWIN(drift_delta)
        self.predicted_labels = deque(maxlen=kappa_window)
        self.kappa = -sys.maxsize
        self.kappa_window = kappa_window

    def update_kappa(self, actual_labels):
        if len(self.predicted_labels) < self.kappa_window:
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

class Pearl:

    def __init__(self,
                 num_trees,
                 repo_size,
                 edit_distance_threshold,
                 bg_kappa_threshold,
                 cd_kappa_threshold,
                 kappa_window,
                 lossy_window_size,
                 reuse_window_size,
                 reuse_rate_upper_bound,
                 warning_delta,
                 drift_delta,
                 arf_max_features,
                 enable_state_adaption=True,
                 enable_state_graph=True,
                 logger=None):

        self.num_trees = num_trees
        self.warning_delta = warning_delta
        self.drift_delta = drift_delta
        self.repo_size = repo_size
        self.edit_distance_threshold = edit_distance_threshold
        self.bg_kappa_threshold = bg_kappa_threshold
        self.cd_kappa_threshold = cd_kappa_threshold
        self.kappa_window = kappa_window
        self.arf_max_features = arf_max_features

        self.enable_state_adaption = enable_state_adaption
        self.enable_state_graph = enable_state_graph
        self.logger = logger

        self.adaptive_trees = [AdaptiveTree(tree_pool_id=i,
                                            tree=ARFHoeffdingTree(max_features=arf_max_features),
                                            kappa_window=kappa_window,
                                            warning_delta=warning_delta,
                                            drift_delta=drift_delta
                          ) for i in range(0, num_trees)]
    
        self.cur_state = ['1' if i < num_trees else '0' for i in range(0, repo_size)]
    
        self.lru_states = LRU_state(capacity=repo_size, edit_distance_threshold=edit_distance_threshold)
        self.lru_states.enqueue(self.cur_state)
    
        self.state_graph = LossyStateGraph(repo_size, lossy_window_size)
    
        self.graph_switch = GraphSwitch(window_size=reuse_window_size,
                                   state_graph=self.state_graph,
                                   reuse_rate=reuse_rate_upper_bound)
    
        self.tree_pool = [None] * repo_size
        for i in range(0, num_trees):
            self.tree_pool[i] = self.adaptive_trees[i]

    def update_drift_detector(self, adaptive_tree, predicted_label, actual_label):
        if predicted_label == actual_label:
            adaptive_tree.warning_detector.add_element(0)
            adaptive_tree.drift_detector.add_element(0)
        else:
            adaptive_tree.warning_detector.add_element(1)
            adaptive_tree.drift_detector.add_element(1)
    
    def predict(self, X, y, adaptive_trees, should_vote):
        predictions = []
    
        for i in range(0, len(X)):
            feature_row = X[i]
            label = y[i]
    
            votes = defaultdict(int)
            for adaptive_tree in adaptive_trees:
                predicted_label = adaptive_tree.tree.predict([feature_row])[0]
    
                adaptive_tree.predicted_labels.append(predicted_label) # for kappa calculation
                if should_vote:
                    self.update_drift_detector(adaptive_tree, predicted_label, label)
    
                votes[predicted_label] += 1
    
                # background tree needs to predict for performance measurement
                if adaptive_tree.bg_adaptive_tree is not None:
                    self.predict([feature_row], [label], [adaptive_tree.bg_adaptive_tree], False)
    
            if should_vote:
                predictions.append(max(votes, key=votes.get))
    
        return predictions
    
    def partial_fit(self, X, y):
        for i in range(0, len(X)):
            for adaptive_tree in self.adaptive_trees:
                n = np.random.poisson(1)
                adaptive_tree.tree.partial_fit([X[i]], [y[i]], sample_weight=[n])
                if adaptive_tree.bg_adaptive_tree is not None:
                    adaptive_tree.bg_adaptive_tree.tree.partial_fit([X[i]], [y[i]], sample_weight=[n])
    
    def update_candidate_trees(self,
                               candidate_trees,
                               closest_state,
                               cur_tree_pool_size):
        if len(closest_state) == 0:
            return
    
        # print(f"closest_state {closest_state}")
        # print(f"cur_state {cur_state}")
    
        for i in range(0, cur_tree_pool_size):
    
            if self.cur_state[i] == '0' \
                    and closest_state[i] == '1' \
                    and not self.tree_pool[i].is_candidate:
    
                if len(candidate_trees) >= self.num_trees:
                    worst_candidate = candidate_trees.pop(0)
                    worst_candidate.reset()
    
                self.tree_pool[i].is_candidate = True
                candidate_trees.append(self.tree_pool[i])
    
        # print("candidate_trees", [c.tree_pool_id for c in candidate_trees])
    
    def select_candidate_trees(self,
                               count,
                               target_state,
                               warning_tree_id_list,
                               candidate_trees,
                               cur_tree_pool_size):
    
        if self.enable_state_graph:
            # try trigger lossy counting
            if self.state_graph.update(len(warning_tree_id_list)):
                self.logger.info(f"{count},lossy counting triggered")
    
        if self.state_graph.is_stable:
            for warning_tree_id in warning_tree_id_list:
                # print("finding next_id...")
                next_id = self.state_graph.get_next_tree_id(warning_tree_id)
                if next_id == -1:
                    # print(f"tree {warning_tree_id} does not have next id")
                    self.state_graph.is_stable = False
    
                else:
                    # print("Next tree found, adding candidate tree...")
                    if not self.tree_pool[next_id].is_candidate:
                        candidate_trees.append(self.tree_pool[next_id])
                        # print("candidate tree added")
    
        if not self.state_graph.is_stable:
            self.logger.info(f"{count},pattern matching")
    
            # trigger pattern matching
            closest_state = self.lru_states.get_closest_state(target_state)
    
            self.update_candidate_trees(candidate_trees=candidate_trees,
                                        closest_state=closest_state,
                                        cur_tree_pool_size=cur_tree_pool_size)
        else:
            self.logger.info(f"{count},graph transition")
    
    def adapt_state(self,
                    drifted_tree_list,
                    candidate_trees,
                    cur_tree_pool_size,
                    drifted_tree_pos,
                    actual_labels):
    
        # sort candidates by kappa
        for candidate_tree in candidate_trees:
            candidate_tree.update_kappa(actual_labels)
        candidate_trees.sort(key=lambda c : c.kappa)
    
        for drifted_tree in drifted_tree_list:
            # TODO
            if cur_tree_pool_size >= self.repo_size:
                print("early break")
                exit()
    
            drifted_tree.update_kappa(actual_labels)
            swap_tree = drifted_tree
    
            background_count = 0
            candidate_count = 0
    
            if len(candidate_trees) > 0 \
                    and candidate_trees[-1].kappa - drifted_tree.kappa >= self.cd_kappa_threshold:
                # swap drifted tree with the candidate tree
                swap_tree = candidate_trees.pop()
                swap_tree.is_candidate = False
    
                # candidate_count += 1
                if self.enable_state_graph:
                    self.graph_switch.update(1)
    
            if swap_tree is drifted_tree:
                add_to_repo = True
                # background_count += 1
                if self.enable_state_graph:
                    self.graph_switch.update(0)
    
                if drifted_tree.bg_adaptive_tree is None:
                        swap_tree = \
                            AdaptiveTree(tree=ARFHoeffdingTree(max_features=self.arf_max_features),
                                         kappa_window=self.kappa_window,
                                         warning_delta=self.warning_delta,
                                         drift_delta=self.drift_delta)
    
                else:
                    prediction_win_size = len(drifted_tree.bg_adaptive_tree.predicted_labels)
                    # print(f"bg_tree window size: {prediction_win_size}")
    
                    drifted_tree.bg_adaptive_tree.update_kappa(actual_labels)
                    # print(f"bg_tree kappa: {drifted_tree.bg_adaptive_tree.kappa} "
                    #       f"swap_tree.kappa: {swap_tree.kappa}")
    
                    if drifted_tree.bg_adaptive_tree.kappa == -sys.maxsize:
                        # add bg_adaptive_tree to the repo even if it didn't fill the window
                        swap_tree = drifted_tree.bg_adaptive_tree
    
                    elif drifted_tree.bg_adaptive_tree.kappa - swap_tree.kappa >= self.bg_kappa_threshold:
                        swap_tree = drifted_tree.bg_adaptive_tree
    
                    else:
                        # false positive
                        add_to_repo = False
    
                if add_to_repo:
                    swap_tree.reset()
    
                    # assign a new tree_pool_id for background tree
                    # and add background tree to tree_pool
                    swap_tree.tree_pool_id = cur_tree_pool_size
                    self.tree_pool[cur_tree_pool_size] = swap_tree
                    cur_tree_pool_size += 1
    
            self.cur_state[drifted_tree.tree_pool_id] = '0'
            self.cur_state[swap_tree.tree_pool_id] = '1'
    
            if self.enable_state_graph:
                self.state_graph.add_edge(drifted_tree.tree_pool_id, swap_tree.tree_pool_id)
    
            # replace drifted tree with swap tree
            pos = drifted_tree_pos.pop()
            self.adaptive_trees[pos] = swap_tree
            drifted_tree.reset()
    
        if self.enable_state_graph:
            self.graph_switch.switch()
    
        return cur_tree_pool_size
    
    def prequential_evaluation(self,
                               stream,
                               max_samples,
                               wait_samples,
                               sample_freq,
                               metric_output_file):
        correct = 0
        x_axis = []
        accuracy_list = []
        actual_labels = deque(maxlen=self.kappa_window) # a window of size kappa_window
    
        sample_counter = 0
        sample_counter_interval = 0
        window_accuracy = 0.0
        window_kappa = 0.0
        window_actual_labels = []
        window_predicted_labels = []
    
        current_state = []
        candidate_trees = []
    
        cur_tree_pool_size = self.num_trees
    
        with open(metric_output_file, 'w') as out:
            out.write("count,accuracy,kappa,memory\n")
    
            for count in range(0, max_samples):
                X, y = stream.next_sample()
                actual_labels.append(y[0])
    
                # test
                prediction = self.predict(X, y, self.adaptive_trees, should_vote=True)[0]
    
                # test on candidate trees
                self.predict(X, y, candidate_trees, should_vote=False)
    
                window_actual_labels.append(y[0])
                window_predicted_labels.append(prediction)
                if prediction == y[0]:
                    correct += 1
    
                target_state = copy.deepcopy(self.cur_state)
    
                warning_tree_id_list = []
                drifted_tree_list = []
                drifted_tree_pos = []
    
                for i in range(0, self.num_trees):
    
                    tree = self.adaptive_trees[i]
                    warning_detected_only = False
                    if tree.warning_detector.detected_change():
                        warning_detected_only = True
                        tree.warning_detector.reset()
    
                        tree.bg_adaptive_tree = \
                            AdaptiveTree(tree=ARFHoeffdingTree(max_features=self.arf_max_features),
                                         kappa_window=self.kappa_window,
                                         warning_delta=self.warning_delta,
                                         drift_delta=self.drift_delta)
    
                    if tree.drift_detector.detected_change():
                        warning_detected_only = False
                        tree.drift_detector.reset()
                        drifted_tree_list.append(tree)
                        drifted_tree_pos.append(i)
    
                        if not self.enable_state_adaption:
                            if tree.bg_adaptive_tree is None:
                                tree.tree = ARFHoeffdingTree(max_features=self.arf_max_features)
                            else:
                                tree.tree = tree.bg_adaptive_tree.tree
                            tree.reset()
    
                    if warning_detected_only:
                        warning_tree_id_list.append(tree.tree_pool_id)
                        target_state[tree.tree_pool_id] = '2'
    
                if self.enable_state_adaption:
                    # if warnings are detected, find closest state and update candidate_trees list
                    if len(warning_tree_id_list) > 0:
                        self.select_candidate_trees(count=count,
                                                    target_state=target_state,
                                                    warning_tree_id_list=warning_tree_id_list,
                                                    candidate_trees=candidate_trees,
                                                    cur_tree_pool_size=cur_tree_pool_size)
    
                    # if actual drifts are detected, swap trees and update cur_state
                    if len(drifted_tree_list) > 0:
                        cur_tree_pool_size = self.adapt_state(drifted_tree_list=drifted_tree_list,
                                                              candidate_trees=candidate_trees,
                                                              cur_tree_pool_size=cur_tree_pool_size,
                                                              drifted_tree_pos=drifted_tree_pos,
                                                              actual_labels=actual_labels)
    
                    self.lru_states.enqueue(self.cur_state)
    
                if (count % wait_samples == 0) and (count != 0):
                    accuracy = correct / wait_samples
    
                    window_accuracy = (window_accuracy * sample_counter + accuracy) \
                        / (sample_counter + 1)
    
                    kappa = cohen_kappa_score(window_actual_labels, window_predicted_labels)
                    window_kappa = (window_kappa * sample_counter + kappa) \
                            / (sample_counter + 1)
    
                    sample_counter += 1
                    sample_counter_interval += wait_samples
                    correct = 0
    
                    if sample_counter_interval == sample_freq:
                        x_axis.append(count)
                        accuracy_list.append(window_accuracy)
    
                        memory_usage = 0
                        if self.enable_state_adaption:
                            memory_usage = self.lru_states.get_size()
                        if self.enable_state_graph:
                            memory_usage += self.state_graph.get_size()
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
                self.partial_fit(X, y)
    
        print(f"length of candidate_trees: {len(candidate_trees)}")

