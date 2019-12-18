import copy
from collections import deque

import numpy as np
from sklearn.metrics import cohen_kappa_score

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
    
        with open(metric_output_file, 'w') as out:
            out.write("count,accuracy,kappa,memory\n")
    
            for count in range(0, max_samples):
                X, y = stream.next_sample()
                actual_labels.append(y[0])
    
                # test
                prediction = classifier.predict(X, y, classifier.adaptive_trees, should_vote=True)[0]

		# test on candidate trees
                classifier.predict(X, y, classifier.candidate_trees, should_vote=False)

                window_actual_labels.append(y[0])
                window_predicted_labels.append(prediction)
                if prediction == y[0]:
                    correct += 1

                classifier.handle_drift(count, actual_labels)
    
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
    
        print(f"length of candidate_trees: {len(classifier.candidate_trees)}")
