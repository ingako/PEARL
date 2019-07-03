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
fig, ax = plt.subplots(3, 2) # sharex=True, sharey=True)

max_samples = 20000
n_wait = 1000
pretrain_size = n_wait

# for computing hoeffding bound
class_count = 2
r = math.log(class_count, 2)
confidence_intervals = [0.05, 0.1, 0.2]

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

    hoeffding_results = []
    for ci in confidence_intervals:
        hoeffding_bound = compute_hoeffding_bound(r, ci, n_wait)
        if abs(error_rate - drift_error_rate) < hoeffding_bound:
            hoeffding_results.append("within boundary")
        else:
            hoeffding_results.append("out of boundary")

    print(f"{count},{hoeffding_results[0]},{hoeffding_results[1]},{hoeffding_results[2]}")

    return ks_pvalue, kappa, fisher_pvalue

def prepare_stream(noise_1 = 0.1, noise_2 = 0.1):
    stream_1 = LEDGeneratorDrift(random_state=0,
                                 noise_percentage=noise_1,
                                 has_noise=False,
                                 n_drift_features=0)

    stream_2 = LEDGeneratorDrift(random_state=0,
                                 noise_percentage=noise_2,
                                 has_noise=False,
                                 n_drift_features=5)

    stream = ConceptDriftStream(
        stream=stream_1,
        drift_stream=stream_2,
        random_state=None,
        position=10000,
        width=1)

    stream.prepare_for_use()
    return stream

noise_levels = [0.2, 0.3, 0.4]
print("#instances,ci=0.05,ci=0.1,ci=0.2")

for i in range(0, len(noise_levels)):
    print(f"#noise={noise_levels[i]}")

    accuracy_list_old = []
    accuracy_list_new = []

    ks_pvalues = []
    kappa_values = []
    fisher_pvalues = []

    learner = HoeffdingTree()
    drift_learner = HoeffdingTree()
    adwin = ADWIN()
    stream = prepare_stream(noise_2=noise_levels[i])

    if pretrain_size > 0:
        X, y = stream.next_sample(pretrain_size)
        learner.partial_fit(X, y)

    correct = 0
    drift_correct = 0

    drift_detected = False

    predictions = []
    drift_predictions = []

    x_axis_old = []
    x_axis_new = []

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
            # print(f"Drift detected at {count}")

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

            x_axis_old.append(count)
            accuracy_list_old.append(accuracy)

            if drift_detected:
                x_axis_new.append(count)
                accuracy_list_new.append(drift_accuracy)

                ks_pvalue, kappa, fisher_pvalue = stats_test(correct, drift_correct, accuracy, drift_accuracy)
                ks_pvalues.append(ks_pvalue)
                kappa_values.append(kappa)
                fisher_pvalues.append(fisher_pvalue)
            # else:
                # print(f"{count},{accuracy}")

            # reset all metrics
            correct = 0
            drift_correct = 0
            predictions = []
            drift_predictions = []


    ax[i, 0].set_title(r"noise=%s" % str(noise_levels[i]))
    ax[i, 0].plot(x_axis_old, accuracy_list_old)
    ax[i, 0].plot(x_axis_new, accuracy_list_new)

    ax[0, 1].plot(x_axis_new, ks_pvalues, label=r"noise=%s" % str(noise_levels[i]))
    ax[1, 1].plot(x_axis_new, kappa_values, label=r"noise=%s" % str(noise_levels[i]))
    ax[2, 1].plot(x_axis_new, fisher_pvalues, label=r"noise=%s" % str(noise_levels[i]))

plt.xlabel("no. of instances")
# plt.ylabel("")

ax[0, 1].set_title("KS Test p-value")
ax[1, 1].set_title("Kappa")
ax[2, 1].set_title("Fisher's Exact Test p-values")

ax[1, 1].legend()

plt.tight_layout()
plt.show()
