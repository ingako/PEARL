#!/usr/bin/env python3

from skmultiflow.trees import HoeffdingTree
from skmultiflow.drift_detection.adwin import ADWIN

from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data import LEDGeneratorDrift
from skmultiflow.data import ConceptDriftStream

from scipy.stats import fisher_exact

import matplotlib
matplotlib.rcParams["backend"] = "Qt4Agg"

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

max_samples = 20000
pretrain_size = 200
n_wait = 200

learner = HoeffdingTree()
learner_drift = HoeffdingTree()
adwin = ADWIN()
stream = prepare_stream(noise_2=0.3)

if pretrain_size > 0:
    X, y = stream.next_sample(pretrain_size)
    learner.partial_fit(X, y)

count = 0
correct = 0

drift_detected = False
drift_correct = 0

contingency_table = [[0]*2 for i in range(2)]

for count in range(0, max_samples):

    X, y = stream.next_sample()

    # test
    if learner.predict(X)[0] == y[0]:
        correct += 1
        adwin.add_element(0)
    else:
        adwin.add_element(1)

    if drift_detected and learner_drift.predict(X)[0] == y[0]:
        drift_correct += 1

    if adwin.detected_change():
        drift_detected = True
        contingency_table = [[0]*2 for i in range(2)]
        print(f"Drift detected at {count}")

    # train
    learner.partial_fit(X, y)
    if drift_detected:
        learner_drift.partial_fit(X, y)

    if (count % n_wait == 0) and (count != 0):
        # log metrics
        accuracy = correct / n_wait
        drift_accuracy = drift_correct / n_wait
        print(f"{count},{accuracy},{drift_accuracy}")

        if drift_detected:
            contingency_table[0][0] += correct;
            contingency_table[0][1] += n_wait - correct;

            contingency_table[1][0] += drift_correct;
            contingency_table[1][1] += n_wait - drift_correct;

        correct = 0
        drift_correct = 0

oddsratio, pvalue = fisher_exact(contingency_table)
print(contingency_table)
print(f"fisher's exact: {pvalue}")
