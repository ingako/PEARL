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
from scipy.stats import anderson_ksamp
from scipy.stats import fisher_exact

from sklearn.metrics import cohen_kappa_score

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["backend"] = "Qt4Agg"
plt.rcParams["figure.figsize"] = (20, 10)

generator = sys.argv[1]

func = int(sys.argv[2])
alt_func = int(sys.argv[3])

drift_width = int(sys.argv[4])
plot_hoeffding = sys.argv[5]

filename = f"{generator}-{func}-{alt_func}-w{drift_width}-{plot_hoeffding}"
print(f"Running on {generator} {func}-{alt_func} with drift width {drift_width}")

if plot_hoeffding == 'f':
    fig, ax = plt.subplots(3, 5, sharey=True, constrained_layout=True)
else:
    fig, ax = plt.subplots(3, 4, sharey=True, constrained_layout=True) # sharex=True

max_samples = 10000
n_wait = 200
pretrain_size = n_wait * 5
p_value = 0.05

# for computing hoeffding bound
class_count = 2
r = math.log(class_count, 2)
confidence_intervals = [0.05, 0.1, 0.2]

def plot_stats(ax, x_axis, y_axis, bound, alt_color='C0'):
    for xy in zip(x_axis, y_axis):
        color=alt_color
        if xy[1] < bound:
            color = 'C3'
        ax.plot(xy[0], xy[1], 'o', color=color, picker=True)

def compute_hoeffding_bound(r, confidence, n_samples):
    # return math.sqrt(((r*r) * math.log(1.0/confidence)) / (2.0*n_samples));
    return np.sqrt((r * r * np.log(1.0/confidence)) / (2.0*n_samples))

def hoeffding_test(correct, drift_correct, hoeffding_results):
    incorrect = n_wait - correct;
    drift_incorrect = n_wait - drift_correct;

    # hoeffding test
    error_rate = incorrect / n_wait
    drift_error_rate = drift_incorrect / n_wait

    for i in range(0, len(confidence_intervals)):
        ci = confidence_intervals[i]
        hoeffding_bound = compute_hoeffding_bound(r, ci, n_wait)
        if abs(error_rate - drift_error_rate) < hoeffding_bound:
            hoeffding_results[i].append(1)
        else:
            hoeffding_results[i].append(0)

def stats_test(correct, drift_correct):
    incorrect = n_wait - correct;
    drift_incorrect = n_wait - drift_correct;
    contingency_table = [[correct, incorrect], [drift_correct, drift_incorrect]]

    # KS test
    # start = timer()
    _, ks_pvalue = ks_2samp(predictions, drift_predictions)
    # end = timer()
    # print(f"ks time: {end - start}")

    # anderson-daring
    # start = timer()
    _, _, ad_stat = anderson_ksamp([predictions, drift_predictions])
    # end = timer()
    # print(f"anderson time: {end - start}")

    # fisher's exact
    _, fisher_pvalue = fisher_exact(contingency_table)

    # kappa statistics
    kappa = cohen_kappa_score(predictions, drift_predictions)

    return ad_stat, ks_pvalue, kappa, fisher_pvalue


noise_levels = [0.2, 0.3, 0.35]
print("#instances,ci=0.05,ci=0.1,ci=0.2")

for i in range(0, len(noise_levels)):
    print(f"#noise={noise_levels[i]}")

    accuracy_list_old = []
    accuracy_list_new = []
    hoeffding_results = [[], [], []]

    ad_stats = []
    ks_pvalues = []
    kappa_values = []
    fisher_pvalues = []

    learner = HoeffdingTree()
    drift_learner = HoeffdingTree()
    warning = ADWIN(0.001)
    adwin = ADWIN(0.0001)
    change_detector_enabled = True

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
    backtrack_counter = 0
    backtrack_limit = n_wait * 2

    for count in range(0, max_samples):

        X, y = stream.next_sample()

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
            print(f"Warning detected at {count}")

        if change_detector_enabled and adwin.detected_change():
            drift_detected = True
            print(f"Drift detected at {count}")

            ax[i, 0].axvline(x=count, color='k', linestyle='--')
            ax[i, 0].text(count + 100, 0.9, f"drift={count}", fontsize=12)

            predictions = []
            drift_predictions = []
            adwin.reset()
            change_detector_enabled = False

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

                if plot_hoeffding == 'f':
                    ad_stat, ks_pvalue, kappa, fisher_pvalue = stats_test(correct,
                                                             drift_correct)
                    ad_stats.append(ad_stat)
                    ks_pvalues.append(ks_pvalue)
                    kappa_values.append(kappa)
                    fisher_pvalues.append(fisher_pvalue)
                else:
                    hoeffding_test(correct, drift_correct, hoeffding_results)

            # reset all metrics
            correct = 0
            drift_correct = 0
            predictions = []
            drift_predictions = []

    ax[i, 0].set_title(r"noise=%s" % str(noise_levels[i]))
    ax[i, 0].plot(x_axis_old, accuracy_list_old)
    ax[i, 0].plot(x_axis_new, accuracy_list_new)

    if plot_hoeffding == 'f':
        plot_stats(ax[i, 1], x_axis_new, ks_pvalues, bound=p_value)
        plot_stats(ax[i, 2], x_axis_new, fisher_pvalues, bound=p_value)
        ax[i, 3].plot(x_axis_new, kappa_values, 'o', color='C2')#, label=r"noise=%s" % str(noise_levels[i]))
        plot_stats(ax[i, 4], x_axis_new, ad_stats, bound=p_value)
    else:
        for ci_idx in range(0, len(confidence_intervals)):
            plot_stats(ax[i, ci_idx + 1], x_axis_new, hoeffding_results[ci_idx], bound=0.5, alt_color='C4')

if plot_hoeffding == 'f':
    ax[0, 1].set_title("KS Test p-value")
    ax[0, 2].set_title("Fisher's Exact Test p-values")
    ax[0, 3].set_title("Kappa")
    ax[0, 4].set_title("Anderson")
else:
    for i in range(0, len(confidence_intervals)):
        # plot_stats(ax[0, i+1], x_axis_new, hoeffding_results[1], bound=0.5, alt_color='C4')
        ax[0, i+1].set_title(f"Hoeffding Bound ci={confidence_intervals[i]}")

# ax[0, 4].legend()

plt.xlabel("no. of instances")
# plt.ylabel("")

#plt.tight_layout()
plt.savefig(f'images/{filename}.eps', format='eps', dpi=1000)
plt.show()
