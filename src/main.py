#!/usr/bin/env python

import argparse
import math
import random
import pathlib
import time
import logging
import os.path

import numpy as np
from skmultiflow.data.file_stream import FileStream

from evaluator import Evaluator
from pearl import Pearl

import sys
path = r'../'
if path not in sys.path:
    sys.path.append(path)
from build.pearl import adaptive_random_forest, pearl

formatter = logging.Formatter('%(message)s')

def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp",
                        dest="cpp", action="store_true",
                        help="Enable cpp backend")
    parser.set_defaults(cpp=False)

    # real world datasets
    parser.add_argument("--dataset_name",
                        dest="dataset_name", default="", type=str,
                        help="dataset name")
    parser.add_argument("--data_format",
                        dest="data_format", default="", type=str,
                        help="dataset format {csv|arff}")

    # pre-generated synthetic datasets
    parser.add_argument("-g", "--is_generated_data",
                        dest="is_generated_data", action="store_true",
                        help="Handle dataset as pre-generated synthetic dataset")
    parser.set_defaults(is_generator_data=False)
    parser.add_argument("--generator_name",
                        dest="generator_name", default="agrawal", type=str,
                        help="name of the synthetic data generator")
    parser.add_argument("--generator_traits",
                        dest="generator_traits", default="abrupt/0", type=str,
                        help="Traits of the synthetic data")
    parser.add_argument("--generator_seed",
                        dest="generator_seed", default=0, type=int,
                        help="Seed used for generating synthetic data")

    # pearl params
    parser.add_argument("-t", "--tree",
                        dest="num_trees", default=60, type=int,
                        help="number of trees in the forest")
    parser.add_argument("-c", "--candidate_tree",
                        dest="max_num_candidate_trees", default=60, type=int,
                        help="max number of candidate trees in the forest")
    parser.add_argument("--state_queue_size",
                        dest="state_queue_size", default=10000000, type=int,
                        help="size of the LRU state queue")
    parser.add_argument("--repo_size",
                        dest="repo_size", default=96000, type=int,
                        help="number of trees in the online tree repository")
    parser.add_argument("-w", "--warning",
                        dest="warning_delta", default=0.0001, type=float,
                        help="delta value for drift warning detector")
    parser.add_argument("-d", "--drift",
                        dest="drift_delta", default=0.00001, type=float,
                        help="delta value for drift detector")
    parser.add_argument("--max_samples",
                        dest="max_samples", default=200000, type=int,
                        help="total number of samples")
    parser.add_argument("--sample_freq",
                        dest="sample_freq", default=1000, type=int,
                        help="log interval for performance")
    parser.add_argument("--kappa_window",
                        dest="kappa_window", default=50, type=int,
                        help="number of instances must be seen for calculating kappa")
    parser.add_argument("--poisson_lambda",
                        dest="poisson_lambda", default=6, type=int,
                        help="lambda for poisson distribution")
    parser.add_argument("--random_state",
                        dest="random_state", default=0, type=int,
                        help="Seed used for adaptive hoeffding tree")

    # tree params
    parser.add_argument("--grace_period",
                        dest="grace_period", default=200, type=int,
                        help="grace period")
    parser.add_argument("--split_confidence",
                        dest="split_confidence", default=0.0000001, type=float,
                        help="split confidence")
    parser.add_argument("--tie_threshold",
                        dest="tie_threshold", default=0.05, type=float,
                        help="tie threshold")
    parser.add_argument("--binary_splits",
                        dest="binary_splits", action="store_true",
                        help="Enable binary splits")
    parser.set_defaults(binary_splits=False)
    parser.add_argument("--no_pre_prune",
                        dest="no_pre_prune", action="store_true",
                        help="Enable no pre prune")
    parser.set_defaults(no_pre_prune=False)
    parser.add_argument("--nb_threshold",
                        dest="nb_threshold", default=0, type=int,
                        help="nb threshold")
    parser.add_argument("--leaf_prediction_type",
                        dest="leaf_prediction_type", default=0, type=int,
                        help="0=MC, 1=NB, 2=NBAdaptive")

    parser.add_argument("-s", "--enable_state_adaption",
                        dest="enable_state_adaption", action="store_true",
                        help="enable the state adaption algorithm")
    parser.set_defaults(enable_state_adaption=False)
    parser.add_argument("-p", "--enable_state_graph",
                        dest="enable_state_graph", action="store_true",
                        help="enable state transition graph")
    parser.set_defaults(enable_state_graph=False)

    parser.add_argument("--cd_kappa_threshold",
                        dest="cd_kappa_threshold", default=0.2, type=float,
                        help="Kappa value that the candidate tree needs to outperform both"
                             "background tree and foreground drifted tree")
    parser.add_argument("--bg_kappa_threshold",
                        dest="bg_kappa_threshold", default=0.00, type=float,
                        help="Kappa value that the background tree needs to outperform the "
                             "foreground drifted tree to prevent from false positive")
    parser.add_argument("--edit_distance_threshold",
                        dest="edit_distance_threshold", default=100, type=int,
                        help="The maximum edit distance threshold")
    parser.add_argument("--lossy_window_size",
                        dest="lossy_window_size", default=5, type=int,
                        help="Window size for lossy count")
    parser.add_argument("--reuse_window_size",
                        dest="reuse_window_size", default=0, type=int,
                        help="Window size for calculating reuse rate")
    parser.add_argument("--reuse_rate_upper_bound",
                        dest="reuse_rate_upper_bound", default=0.4, type=float,
                        help="The reuse rate threshold for switching from "
                             "pattern matching to graph transition")
    parser.add_argument("--reuse_rate_lower_bound",
                        dest="reuse_rate_lower_bound", default=0.1, type=float,
                        help="The reuse rate threshold for switching from "
                             "pattern matching to graph transition")

    args = parser.parse_args()

    # TODO
    if args.reuse_rate_upper_bound < args.reuse_rate_lower_bound:
        exit("reuse rate upper bound must be greater than or equal to the lower bound")

    if args.enable_state_graph:
        args.enable_state_adaption = True


    # prepare data
    if args.is_generated_data:
        data_file_dir = f"../data/{args.generator_name}/" \
                        f"{args.generator_traits}/"
        data_file_path = f"{data_file_dir}/{args.generator_seed}.{args.data_format}"
        result_directory = f"{args.generator_name}/{args.generator_traits}/"

    else:
        data_file_dir = f"../data/" \
                         f"{args.dataset_name}/"
        data_file_path = f"{data_file_dir}/{args.dataset_name}.{args.data_format}"
        result_directory = args.dataset_name

    if not os.path.isfile(data_file_path):
        print(f"Cannot locate file at {data_file_path}")
        exit()

    print(f"Preparing stream from file {data_file_path}...")

    if args.cpp:
        print("speed optimization with C++")
        stream = data_file_path
    else:
        print(f"PEARL python: preparing stream from file {data_file_path}...")
        stream = FileStream(data_file_path)
        stream.prepare_for_use()
        args.max_samples = min(args.max_samples, stream.n_remaining_samples())


    if args.enable_state_graph:
        result_directory = f"{result_directory}/" \
                           f"k{args.cd_kappa_threshold}-e{args.edit_distance_threshold}/" \
                           f"r{args.reuse_rate_upper_bound}-r{args.reuse_rate_lower_bound}-" \
                           f"w{args.reuse_window_size}/" \
                           f"lossy-{args.lossy_window_size}"

    elif args.enable_state_adaption:
        result_directory = f"{result_directory}/" \
                           f"k{args.cd_kappa_threshold}-e{args.edit_distance_threshold}/"

    pathlib.Path(result_directory).mkdir(parents=True, exist_ok=True)

    metric_output_file = f"{result_directory}/result-{args.generator_seed}.csv"


    configs = (
        f"metric_output_file: {metric_output_file}\n"
        f"warning_delta: {args.warning_delta}\n"
        f"drift_delta: {args.drift_delta}\n"
        f"max_samples: {args.max_samples}\n"
        f"sample_freq: {args.sample_freq}\n"
        f"kappa_window: {args.kappa_window}\n"
        f"random_state: {args.random_state}\n"
        f"enable_state_adaption: {args.enable_state_adaption}\n"
        f"enable_state_graph: {args.enable_state_graph}\n")

    print(configs)
    with open(f"{result_directory}/config", 'w') as out:
        out.write(configs)
        out.flush()

    if args.cpp:
        arf_max_features = -1
        num_features = -1
    else:
        num_features = stream.n_features
        arf_max_features = int(math.log2(num_features)) + 1

    np.random.seed(args.random_state)
    random.seed(0)

    if args.enable_state_adaption:
        with open(f"{result_directory}/reuse-rate-{args.generator_seed}.log", 'w') as out:
            out.write("background_window_count,candidate_window_count,reuse_rate\n")

    metrics_logger = setup_logger('metrics', metric_output_file)
    process_logger = setup_logger('process', f'{result_directory}/processes-{args.generator_seed}.info')

    if args.cpp:
        if not args.enable_state_adaption and not args.enable_state_graph:
            pearl = adaptive_random_forest(args.num_trees,
                                           arf_max_features,
                                           args.poisson_lambda,
                                           args.random_state,
                                           args.grace_period,
                                           args.split_confidence,
                                           args.tie_threshold,
                                           args.binary_splits,
                                           args.no_pre_prune,
                                           args.nb_threshold,
                                           args.leaf_prediction_type,
                                           args.warning_delta,
                                           args.drift_delta)
            print("init adaptive_random_forest cpp")

        else:
            pearl = pearl(args.num_trees,
                          args.max_num_candidate_trees,
                          args.repo_size,
                          args.state_queue_size,
                          args.edit_distance_threshold,
                          args.kappa_window,
                          args.lossy_window_size,
                          args.reuse_window_size,
                          arf_max_features,
                          args.poisson_lambda,
                          args.random_state,
                          args.bg_kappa_threshold,
                          args.cd_kappa_threshold,
                          args.reuse_rate_upper_bound,
                          args.warning_delta,
                          args.drift_delta,
                          args.enable_state_adaption,
                          args.enable_state_graph,
                          args.grace_period,
                          args.split_confidence,
                          args.tie_threshold,
                          args.binary_splits,
                          args.no_pre_prune,
                          args.nb_threshold,
                          args.leaf_prediction_type)
            print("init pearl cpp")
        eval_func = Evaluator.prequential_evaluation_cpp

    else:
        pearl = Pearl(num_trees=args.num_trees,
                      repo_size=args.repo_size,
                      edit_distance_threshold=args.edit_distance_threshold,
                      bg_kappa_threshold=args.bg_kappa_threshold,
                      cd_kappa_threshold=args.cd_kappa_threshold,
                      kappa_window=args.kappa_window,
                      lossy_window_size=args.lossy_window_size,
                      reuse_window_size=args.reuse_window_size,
                      reuse_rate_upper_bound=args.reuse_rate_upper_bound,
                      warning_delta=args.warning_delta,
                      drift_delta=args.drift_delta,
                      arf_max_features=arf_max_features,
                      enable_state_adaption=args.enable_state_adaption,
                      enable_state_graph=args.enable_state_graph,
                      logger=process_logger)
        eval_func = Evaluator.prequential_evaluation

    eval_func(classifier=pearl,
              stream=stream,
              max_samples=args.max_samples,
              sample_freq=args.sample_freq,
              metrics_logger=metrics_logger)
