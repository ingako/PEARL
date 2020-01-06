#!/usr/bin/env python3

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
from cpp.pearl import pearl


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

    parser.add_argument("-t", "--tree",
                        dest="num_trees", default=60, type=int,
                        help="number of trees in the forest")
    parser.add_argument("-c", "--candidate_tree",
                        dest="max_num_candidate_trees", default=60, type=int,
                        help="max number of candidate trees in the forest")
    parser.add_argument("-g", "--generator",
                        dest="generator", default="agrawal", type=str,
                        help="name of the synthetic data generator")
    # parser.add_argument("--pool",
    #                     dest="tree_pool_size", default=180, type=int,
    #                     help="number of trees in the online tree repository")
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
    parser.add_argument("--kappa_window",
                        dest="kappa_window", default=50, type=int,
                        help="number of instances must be seen for calculating kappa")
    parser.add_argument("--random_state",
                        dest="random_state", default=0, type=int,
                        help="Seed used for adaptive hoeffding tree")
    parser.add_argument("--generator_seed",
                        dest="generator_seed", default=0, type=int,
                        help="Seed used for generating synthetic data")
    parser.add_argument("--enable_generator_noise",
                        dest="enable_generator_noise", action="store_true",
                        help="Enable noise in synthetic data generator")
    parser.set_defaults(enable_generator_noise=False)

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

    if args.reuse_rate_upper_bound < args.reuse_rate_lower_bound:
        exit("reuse rate upper bound must be greater than or equal to the lower bound")

    if args.enable_state_graph:
        args.enable_state_adaption = True

    stream = None
    potential_file = f"../data/{args.generator}/{args.generator}.csv"
    potential_pre_gen_file = f"../data/{args.generator}/{args.generator}-{args.generator_seed}.csv"

    # prepare data
    if os.path.isfile(potential_file):
        print(f"preparing stream from file {potential_file}...")
        stream = FileStream(potential_file)
        stream.prepare_for_use()

        args.max_samples = min(args.max_samples, stream.n_remaining_samples())

    elif os.path.isfile(potential_pre_gen_file):
        print(f"preparing stream from file {potential_pre_gen_file}...")
        stream = FileStream(potential_pre_gen_file)
        stream.prepare_for_use()

        args.max_samples = min(args.max_samples, stream.n_remaining_samples())

    else:
        print(f"preparing stream from {args.generator} generator...")
        concepts = [4,0,8]
        stream = RecurrentDriftStream(generator=args.generator,
                                      concepts=concepts,
                                      has_noise=args.enable_generator_noise,
                                      random_state=args.generator_seed)
        stream.prepare_for_use()
        print(stream.get_data_info())

    result_directory = args.generator
    if args.enable_generator_noise:
        result_directory = f"{result_directory}-noise"

    metric_output_file = "result"
    time_output_file = "time"

    if args.enable_state_graph:
        result_directory = f"{result_directory}/" \
                           f"k{args.cd_kappa_threshold}-e{args.edit_distance_threshold}/" \
                           f"r{args.reuse_rate_upper_bound}-r{args.reuse_rate_lower_bound}-" \
                           f"w{args.reuse_window_size}/" \
                           f"lossy-{args.lossy_window_size}"

        metric_output_file = f"{metric_output_file}-parf"
        time_output_file = f"{time_output_file}-parf"

    elif args.enable_state_adaption:
        result_directory = f"{result_directory}/" \
                           f"k{args.cd_kappa_threshold}-e{args.edit_distance_threshold}/"

        metric_output_file = f"{metric_output_file}-sarf"
        time_output_file = f"{time_output_file}-sarf"

    pathlib.Path(result_directory).mkdir(parents=True, exist_ok=True)

    metric_output_file = f"{result_directory}/{metric_output_file}-{args.generator_seed}.csv"
    time_output_file = f"{result_directory}/{time_output_file}-{args.generator_seed}.log"


    configs = (
        f"metric_output_file: {metric_output_file}\n"
        f"warning_delta: {args.warning_delta}\n"
        f"drift_delta: {args.drift_delta}\n"
        f"max_samples: {args.max_samples}\n"
        f"wait_samples: {args.wait_samples}\n"
        f"sample_freq: {args.sample_freq}\n"
        f"kappa_window: {args.kappa_window}\n"
        f"random_state: {args.random_state}\n"
        f"enable_state_adaption: {args.enable_state_adaption}\n"
        f"enable_state_graph: {args.enable_state_graph}\n")

    print(configs)
    with open(f"{result_directory}/config", 'w') as out:
        out.write(configs)
        out.flush()

    num_features = stream.n_features
    arf_max_features = int(math.log2(num_features)) + 1

    repo_size = args.num_trees * 160
    np.random.seed(args.random_state)
    random.seed(0)

    if args.enable_state_adaption:
        with open(f"{result_directory}/reuse-rate-{args.generator_seed}.log", 'w') as out:
            out.write("background_window_count,candidate_window_count,reuse_rate\n")

    metrics_logger = setup_logger('metrics', metric_output_file)
    process_logger = setup_logger('process', f'{result_directory}/processes-{args.generator_seed}.info')

    if args.cpp:
        pearl = pearl(args.num_trees,
                      args.max_num_candidate_trees,
                      repo_size,
                      args.edit_distance_threshold,
                      args.kappa_window,
                      args.lossy_window_size,
                      args.reuse_window_size,
                      arf_max_features,
                      args.bg_kappa_threshold,
                      args.cd_kappa_threshold,
                      args.reuse_rate_upper_bound,
                      args.warning_delta,
                      args.drift_delta,
                      args.enable_state_adaption)
        eval_func = Evaluator.prequential_evaluation_cpp

    else:
        pearl = Pearl(num_trees=args.num_trees,
                      repo_size=repo_size,
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

    start = time.process_time()
    eval_func(classifier=pearl,
              stream=stream,
              max_samples=args.max_samples,
              wait_samples=args.wait_samples,
              sample_freq=args.sample_freq,
              metrics_logger=metrics_logger)
    elapsed = time.process_time() - start

    with open(f"{time_output_file}", 'w') as out:
        out.write(str(elapsed) + '\n')
