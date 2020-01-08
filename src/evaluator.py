import copy
from collections import deque

import numpy as np
from sklearn.metrics import cohen_kappa_score

class Evaluator:

    @staticmethod
    def prequential_evaluation(classifier,
                               stream,
                               max_samples,
                               sample_freq,
                               metrics_logger):
        correct = 0
        window_actual_labels = []
        window_predicted_labels = []

        metrics_logger.info("count,accuracy,kappa,memory")

        for count in range(0, max_samples):
            X, y = stream.next_sample()

            # test
            prediction = classifier.predict(X, y)[0]

            window_actual_labels.append(y[0])
            window_predicted_labels.append(prediction)
            if prediction == y[0]:
                correct += 1

            classifier.handle_drift(count)

            if count % sample_freq == 0 and count != 0:

                accuracy = correct / sample_freq
                kappa = cohen_kappa_score(window_actual_labels, window_predicted_labels)
                memory_usage = classifier.get_size()

                print(f"{count},{accuracy},{kappa},{memory_usage}")
                metrics_logger.info(f"{count},{accuracy},{kappa},{memory_usage}")

                correct = 0
                window_actual_labels = []
                window_predicted_labels = []

            # train
            classifier.partial_fit(X, y)

        print(f"length of candidate_trees: {len(classifier.candidate_trees)}")


    @staticmethod
    def prequential_evaluation_cpp(classifier,
                                   stream,
                                   max_samples,
                                   sample_freq,
                                   metrics_logger):
        correct = 0
        window_actual_labels = []
        window_predicted_labels = []

        metrics_logger.info("count,accuracy,candidate_tree_size,tree_pool_size")

        classifier.init_data_source(stream);

        for count in range(0, max_samples):
            if not classifier.get_next_instance():
                break

            correct += 1 if classifier.process() else 0

            if count % sample_freq == 0 and count != 0:
                accuracy = correct / sample_freq
                candidate_tree_size = classifier.get_candidate_tree_group_size()
                tree_pool_size = classifier.get_tree_pool_size()

                print(f"{count},{accuracy},{candidate_tree_size},{tree_pool_size}")
                metrics_logger.info(f"{count},{accuracy}," \
                                    f"{candidate_tree_size},{tree_pool_size}")

                correct = 0
                window_actual_labels = []
                window_predicted_labels = []
