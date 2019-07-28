#!/usr/bin/env python3

import sys
import math
import numpy as np
from timeit import default_timer as timer
from stream_generators import *

from skmultiflow.trees import HoeffdingTree
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.evaluation import EvaluatePrequential
from scipy.stats import ks_2samp

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["backend"] = "Qt4Agg"
plt.rcParams["figure.figsize"] = (20, 10)

generator = sys.argv[1]
func = int(sys.argv[2])
alt_func = int(sys.argv[3])
drift_width = int(sys.argv[4])

filename = f"{generator}-drift-pos-{func}-{alt_func}-w{drift_width}"
print(f"Running on {generator} {func}-{alt_func} with drift width {drift_width}")

fig, ax = plt.subplots(3, 2, sharey=True, constrained_layout=True)

max_samples = 10000
n_wait = 200
pretrain_size = n_wait * 5
p_value = 0.05

# for computing hoeffding bound
class_count = 2
r = math.log(class_count, 2)
# confidence_intervals = [0.05, 0.1, 0.2]

def plot_stats(ax, x_axis, y_axis, bound, alt_color='C0'):
    for xy in zip(x_axis, y_axis):
        color=alt_color
        if xy[1] < bound:
            color = 'C3'
        ax.plot(xy[0], xy[1], 'o', color=color, picker=True)

def stats_test(correct, drift_correct):
    incorrect = n_wait - correct;
    drift_incorrect = n_wait - drift_correct;

    # KS test
    _, ks_pvalue = ks_2samp(predictions, drift_predictions)
    return ks_pvalue

def locate_drift_pos(warning_pos, cached_data_x, cached_data_y, ax):
    data_pos = warning_pos
    exact_drift_pos = warning_pos

    x_axis = []
    accuracy_list_old = []
    accuracy_list_new = []

    while cached_data_x:
        correct = 0
        drift_correct = 0
        interval_len = min(len(cached_data_x), int(n_wait / 4))

        for count in range(0, interval_len):
            x = cached_data_x.pop()
            y = cached_data_y.pop()

            prediction = learner.predict(x)
            if prediction == y:
                correct += 1

            drift_prediction = drift_learner.predict(x)
            if drift_prediction == y:
                drift_correct += 1

        data_pos -= interval_len
        x_axis.append(data_pos)
        accuracy_list_old.append(correct / interval_len)
        accuracy_list_new.append(drift_correct / interval_len)

        if correct < drift_correct:
            exact_drift_pos = data_pos

    reversed(x_axis)
    reversed(accuracy_list_old)
    reversed(accuracy_list_new)

    ax.plot(x_axis, accuracy_list_old)
    ax.plot(x_axis, accuracy_list_new)

    ax.axvline(x=exact_drift_pos, color='r', linestyle='-',
                     label=exact_drift_pos)

    return exact_drift_pos

noise_levels = [0.2, 0.3, 0.35]
print("#instances,ci=0.05,ci=0.1,ci=0.2")

for i in range(0, len(noise_levels)):
    print(f"#noise={noise_levels[i]}")

    accuracy_list_old = []
    accuracy_list_new = []

    ks_pvalues = []

    learner = HoeffdingTree()
    drift_learner = HoeffdingTree()

    warning = ADWIN(0.001)
    if i == 0:
        adwin = ADWIN(0.0001)
    else:
        adwin = ADWIN(0.0000001)
    drift_detector_enabled = True

    if generator == 'led':
        stream_1, stream_2 = prepare_led_streams(noise_2=noise_levels[i], func=func, alt_func=alt_func)
    elif generator == 'sea':
        stream_1, stream_2 = prepare_sea_streams(noise_2=noise_levels[i], func=func, alt_func=alt_func)
    else:
        stream_1, stream_2 = prepare_agrawal_streams(noise_2=noise_levels[i], func=func, alt_func=alt_func)

    stream = prepare_concept_drift_stream(stream_1, stream_2, 5000+pretrain_size,
                                          drift_width)

    if pretrain_size > 0:
        X, y = stream.next_sample(pretrain_size)
        learner.partial_fit(X, y)

    correct = 0
    drift_correct = 0

    warning_detected = False
    drift_detected = False

    predictions = []
    drift_predictions = []

    x_axis_old = []
    x_axis_new = []

    cached_data_x = []
    cached_data_y = []

    for count in range(0, max_samples):

        X, y = stream.next_sample()

        if not warning_detected:
            cached_data_x.append(X)
            cached_data_y.append(y[0])

        # test
        prediction = learner.predict(X)[0]
        predictions.append(prediction)

        if prediction == y[0]:
            correct += 1
            warning.add_element(0)
            adwin.add_element(0)
        else:
            warning.add_element(1)
            adwin.add_element(1)

        if drift_detected:
            drift_prediction = drift_learner.predict(X)[0]
            drift_predictions.append(drift_prediction)

            if drift_prediction == y[0]:
                drift_correct += 1

        if not warning_detected and warning.detected_change():
            warning_detected = True
            warning_pos = count
            print(f"Warning detected at {count}")

            ax[i, 0].axvline(x=count, color='k', linestyle='--', label=str(count))
            # ax[i, 0].text(count + 100, 0.9, f"warning={count}", fontsize=12)

        if drift_detector_enabled and adwin.detected_change():
            drift_detected = True
            print(f"Drift detected at {count}")

            ax[i, 0].axvline(x=count, color='k', linestyle='-', label=str(count))
            # ax[i, 0].text(count + 100, 0.9, f"drift={count}", fontsize=12)

            predictions = []
            drift_predictions = []
            adwin.reset()
            drift_detector_enabled = False

        # train
        if drift_detected:
            drift_learner.partial_fit(X, y)
        else:
            learner.partial_fit(X, y)
            if warning_detected and not drift_detected:
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

                ks_pvalue = stats_test(correct, drift_correct)
                ks_pvalues.append(ks_pvalue)

            # reset all metrics
            correct = 0
            drift_correct = 0
            predictions = []
            drift_predictions = []

    ax[i, 0].set_title(r"noise=%s" % str(noise_levels[i]))
    ax[i, 0].plot(x_axis_old, accuracy_list_old)
    ax[i, 0].plot(x_axis_new, accuracy_list_new)

    # plot_stats(ax[i, 1], x_axis_new, ks_pvalues, bound=p_value)

    # backtrack from the warning position
    exact_drift_pos = locate_drift_pos(warning_pos, cached_data_x, cached_data_y, ax[i, 1])
    ax[i, 0].axvline(x=exact_drift_pos, color='r', linestyle='-',
                     label=exact_drift_pos)
    print(f"Exact drift point found at {exact_drift_pos}")

    ax[i, 0].legend(loc='upper right')

# ax[0, 1].set_title("KS Test p-value")
ax[0, 1].set_title("Backtrack Accuracy")

plt.xlabel("no. of instances")
# plt.ylabel("")

#plt.tight_layout()
plt.savefig(f'images/{filename}.eps', format='eps', dpi=1000)
plt.show()
