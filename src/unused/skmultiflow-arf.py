#!/usr/bin/env python3

import math
import numpy as np
import argparse
from stream_generators import *
from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest
from skmultiflow.drift_detection.adwin import ADWIN

def evaluate():
    stream = RecurrentDriftStream()
    stream.prepare_for_use()
    print(stream.get_data_info())

    learner = AdaptiveRandomForest(n_estimators=args.num_trees,
                                   max_features=arf_max_features,
                                   disable_weighted_vote=True,
                                   lambda_value=1,
                                   # leaf_prediction='mc',
                                   # binary_split=True,
                                   warning_detection_method=ADWIN(args.warning_delta),
                                   drift_detection_method=ADWIN(args.drift_delta),
                                   random_state=np.random)

    correct = 0
    x_axis = []
    accuracy_list = []

    sample_counter = 0
    window_accuracy = 0.0

    with open('results_skarf.csv', 'w') as out:
        out.write("count,accuracy\n")

        for count in range(0, args.max_samples):

            X, y = stream.next_sample()

            # test
            prediction = learner.predict(X)[0]

            if int(prediction) == int(y[0]):
                correct += 1

            if (count % args.wait_samples == 0) and (count != 0):
                accuracy = correct / args.wait_samples
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

            learner.partial_fit(X, y)

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
    np.random.seed(args.random_state)

    evaluate()
