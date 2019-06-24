#!/usr/bin/env python3

from skmultiflow.trees import HoeffdingTree
from skmultiflow.drift_detection.adwin import ADWIN

from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data import LEDGeneratorDrift
from skmultiflow.data import ConceptDriftStream
from matplotlib.figure import Figure

import matplotlib
matplotlib.rcParams["backend"] = "Qt4Agg"

stream_1 = LEDGeneratorDrift(random_state=0,
        noise_percentage=0.1,
        has_noise=True,
        n_drift_features=0)

stream_2 = LEDGeneratorDrift(random_state=0,
        noise_percentage=0.3,
        has_noise=True,
        n_drift_features=0)

stream = ConceptDriftStream(
        stream=stream_1,
        drift_stream=stream_2,
        random_state=None,
        position=10000,
        width=1)

stream.prepare_for_use()

max_samples = 20000
pretrain_size = 200
n_wait = 200

learner = HoeffdingTree()
learner_drift = HoeffdingTree()
adwin = ADWIN()

if pretrain_size > 0:
    X, y = stream.next_sample(pretrain_size)
    learner.partial_fit(X, y)

count = 0
correct = 0

drift_detected = False
drift_correct = 0

while count < max_samples:
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

        correct = 0
        drift_correct = 0

    count += 1
