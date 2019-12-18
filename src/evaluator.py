import copy
from collections import deque

import numpy as np
from sklearn.metrics import cohen_kappa_score
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.trees.arf_hoeffding_tree import ARFHoeffdingTree

from pearl import AdaptiveTree

class Evaluator: 

    @staticmethod
    def prequential_evaluation(classifier,
                               stream,
                               max_samples,
                               wait_samples,
                               sample_freq,
                               metric_output_file):
        correct = 0
        x_axis = []
        accuracy_list = []
        actual_labels = deque(maxlen=classifier.kappa_window) # a window of size kappa_window
    
        sample_counter = 0
        sample_counter_interval = 0
        window_accuracy = 0.0
        window_kappa = 0.0
        window_actual_labels = []
        window_predicted_labels = []
    
        current_state = []
        candidate_trees = []
    
        with open(metric_output_file, 'w') as out:
            out.write("count,accuracy,kappa,memory\n")
    
            for count in range(0, max_samples):
                X, y = stream.next_sample()
                actual_labels.append(y[0])
    
                # test
                prediction = classifier.predict(X, y, classifier.adaptive_trees, should_vote=True)[0]
    
                # test on candidate trees
                classifier.predict(X, y, candidate_trees, should_vote=False)
    
                window_actual_labels.append(y[0])
                window_predicted_labels.append(prediction)
                if prediction == y[0]:
                    correct += 1
    
                target_state = copy.deepcopy(classifier.cur_state)
    
                warning_tree_id_list = []
                drifted_tree_list = []
                drifted_tree_pos = []
    
                for i in range(0, classifier.num_trees):
    
                    tree = classifier.adaptive_trees[i]
                    warning_detected_only = False
                    if tree.warning_detector.detected_change():
                        warning_detected_only = True
                        tree.warning_detector.reset()
    
                        tree.bg_adaptive_tree = \
                            AdaptiveTree(tree=ARFHoeffdingTree(max_features=classifier.arf_max_features),
                                         kappa_window=classifier.kappa_window,
                                         warning_delta=classifier.warning_delta,
                                         drift_delta=classifier.drift_delta)
    
                    if tree.drift_detector.detected_change():
                        warning_detected_only = False
                        tree.drift_detector.reset()
                        drifted_tree_list.append(tree)
                        drifted_tree_pos.append(i)
    
                        if not classifier.enable_state_adaption:
                            if tree.bg_adaptive_tree is None:
                                tree.tree = ARFHoeffdingTree(max_features=classifier.arf_max_features)
                            else:
                                tree.tree = tree.bg_adaptive_tree.tree
                            tree.reset()
    
                    if warning_detected_only:
                        warning_tree_id_list.append(tree.tree_pool_id)
                        target_state[tree.tree_pool_id] = '2'
    
                if classifier.enable_state_adaption:
                    # if warnings are detected, find closest state and update candidate_trees list
                    if len(warning_tree_id_list) > 0:
                        classifier.select_candidate_trees(count=count,
                                                    target_state=target_state,
                                                    warning_tree_id_list=warning_tree_id_list,
                                                    candidate_trees=candidate_trees)
    
                    # if actual drifts are detected, swap trees and update cur_state
                    if len(drifted_tree_list) > 0:
                        classifier.adapt_state(drifted_tree_list=drifted_tree_list,
                                               candidate_trees=candidate_trees,
                                               drifted_tree_pos=drifted_tree_pos,
                                               actual_labels=actual_labels)
    
                    classifier.lru_states.enqueue(classifier.cur_state)
    
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
                        if classifier.enable_state_adaption:
                            memory_usage = classifier.lru_states.get_size()
                        if classifier.enable_state_graph:
                            memory_usage += classifier.state_graph.get_size()
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
                classifier.partial_fit(X, y)
    
        print(f"length of candidate_trees: {len(candidate_trees)}")
