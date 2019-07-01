#!/usr/bin/env python3

import math

from skmultiflow.trees import HoeffdingTree
from skmultiflow.drift_detection.adwin import ADWIN

from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data import LEDGeneratorDrift
from skmultiflow.data import ConceptDriftStream

from scipy.stats import ks_2samp
from scipy.stats import fisher_exact

from sklearn.metrics import cohen_kappa_score

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["backend"] = "Qt4Agg"

max_samples = 20000
n_wait = 1000
pretrain_size = n_wait

# for computing hoeffding bound
class_count = 2
r = math.log(class_count, 2)
confidence = 0.05

def compute_hoeffding_bound(r, confidence, n_samples):
    return math.sqrt(((r*r) * math.log(1.0/confidence)) / (2.0*n_samples));

def stats_test(correct, drift_correct, accuracy, drift_accuracy):
    incorrect = n_wait - correct;
    drift_incorrect = n_wait - drift_correct;
    contingency_table = [[correct, incorrect], [drift_correct, drift_incorrect]]

    # fisher's exact
    _, fisher_pvalue = fisher_exact(contingency_table)

    # kappa statistics
    kappa = cohen_kappa_score(predictions, drift_predictions)

    # KS test
    _, ks_pvalue= ks_2samp(predictions, drift_predictions)

    # hoeffding test
    error_rate = incorrect / n_wait
    drift_error_rate = drift_incorrect / n_wait
    hoeffding_bound = compute_hoeffding_bound(r, confidence, n_wait)
    if abs(error_rate - drift_error_rate) < hoeffding_bound:
        hoeffding_result = "within bound"
    else:
        hoeffding_result = "outside boune"

    print(f"{count},{accuracy},{drift_accuracy}"
          f"{fisher_pvalue},{kappa},{ks_pvalue},{hoeffding_result}")

def prepare_stream(noise_1 = 0.1, noise_2 = 0.1):
    stream_1 = LEDGeneratorDrift(random_state=0,
                                 noise_percentage=noise_1,
                                 has_noise=True,
                                 n_drift_features=0)

    stream_2 = LEDGeneratorDrift(random_state=0,
                                 noise_percentage=noise_2,
                                 has_noise=True,
                                 n_drift_features=0)

    stream = ConceptDriftStream(
        stream=stream_1,
        drift_stream=stream_2,
        random_state=None,
        position=10000,
        width=1)

    stream.prepare_for_use()
    return stream

noise_levels = [0.2, 0.3, 0.4]
for noise_level in noise_levels:
    learner = HoeffdingTree()
    drift_learner = HoeffdingTree()
    adwin = ADWIN()
    stream = prepare_stream(noise_2=noise_level)

    if pretrain_size > 0:
        X, y = stream.next_sample(pretrain_size)
        learner.partial_fit(X, y)

    correct = 0
    drift_correct = 0

    drift_detected = False

    predictions = []
    drift_predictions = []

    for count in range(0, max_samples):

        X, y = stream.next_sample()

        # test
        prediction = learner.predict(X)[0]
        predictions.append(prediction)

        if prediction == y[0]:
            correct += 1
            adwin.add_element(0)
        else:
            adwin.add_element(1)

        if drift_detected:
            drift_prediction = drift_learner.predict(X)[0]
            drift_predictions.append(drift_prediction)

            if drift_prediction == y[0]:
                drift_correct += 1

        if adwin.detected_change():
            drift_detected = True
            print(f"Drift detected at {count}")

            predictions = []
            adwin.reset()

        # train
        learner.partial_fit(X, y)
        if drift_detected:
            drift_learner.partial_fit(X, y)

        if (count % n_wait == 0) and (count != 0):
            # log metrics
            accuracy = correct / n_wait
            drift_accuracy = drift_correct / n_wait

            if drift_detected:
                stats_test(correct, drift_correct, accuracy, drift_accuracy)
            else:
                print(f"{count},{accuracy}")

            # reset all metrics
            correct = 0
            drift_correct = 0
            predictions = []
            drift_predictions = []
