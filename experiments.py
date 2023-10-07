import numpy as np
import sys
import mapreduce
import streaming
import kcenter
import results
import datasets
import itertools
import logging
import signal
from baseline.adapter import KFC

TIMEOUT_SECS = 12*60*60 # 12 hours


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    logging.warning("Timed out!")
    raise TimeoutException()


def evaluate(dataset, delta, algo, k, ofile, shuffle_seed=None):
    if results.already_run(dataset, algo.name(), k, delta, algo.attrs()):
        return
    logging.info(
        f"Running {algo.name()} with {algo.attrs()} on {dataset} for k={k}")
    data, colors, fairness_constraints = datasets.load(
        dataset, 0, delta, shuffle_seed=shuffle_seed)

    try:
        signal.alarm(TIMEOUT_SECS)
        assignment = algo.fit_predict(data, colors, fairness_constraints)
        signal.alarm(0)  # Cancel the alarm
        centers = algo.centers

        results.save_result(ofile, centers, assignment, dataset,
                            algo.name(), k, delta, algo.attrs(),
                            algo.time(), algo.additional_metrics(), shuffle_seed=shuffle_seed)
    except TimeoutException:
        results.save_timeout(dataset, algo.name(), k,
                             delta, algo.attrs(), TIMEOUT_SECS)


def delta_influence():
    ofile = "results.hdf5"
    ks = [32]
    deltas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    all_datasets = datasets.datasets()
    for dataset, delta, k in itertools.product(all_datasets, deltas, ks):
        n, dim = datasets.dataset_size(dataset)
        algos = [
            KFC(k, cplex_path)
        ] + [
            kcenter.CoresetFairKCenter(
                k, tau, cplex_path, seed=seed)
            for tau in [32*k]
            for seed in [1]
            if tau <= n
        ]
        for algo in algos:
            evaluate(dataset, delta, algo, k, ofile)


def exhaustive():
    ofile = "results.hdf5"
    ks = [32, 64]
    # ks = [2, 4, 8, 16, 32]
    deltas = [0.01]
    all_datasets = datasets.datasets()
    all_datasets = ["athlete", "diabetes"]
    for dataset, delta, k in itertools.product(all_datasets, deltas, ks):
        n, dim = datasets.dataset_size(dataset)
        algos = [
            kcenter.Dummy(k),
            kcenter.UnfairKCenter(k),
            kcenter.BeraEtAlKCenter(k, cplex_path),
            KFC(k, cplex_path)
        ] + [
            kcenter.CoresetFairKCenter(
                k, tau, cplex_path, seed=seed)
            #for tau in [2*k, 8*k, 32*k, 64*k, 128*k, 256*k]
            for tau in [32*k]
            for seed in [1,2,3,4,5,6,7]
            if tau <= n
        ]
        for algo in algos:
            evaluate(dataset, delta, algo, k, ofile)


def mr_experiments():
    ofile = "results.hdf5"
    ks = [32]
    deltas = [0.01]
    all_datasets = ["diabetes", "athlete", "census1990", "hmda"]
    for dataset, delta, k in itertools.product(all_datasets, deltas, ks):
        n, dim = datasets.dataset_size(dataset)
        for threads in [2, 4, 8, 16]:
            for shuffle_seed in [11234,1234,4234,1234,432]:#,1,2,3,4,5]:
                algos = [
                    mapreduce.BeraEtAlMRFairKCenter(
                        k, threads, cplex_path, seed=shuffle_seed)
                ] + [
                    mapreduce.MRCoresetFairKCenter(
                        k, tau, threads, cplex_path, seed=shuffle_seed)
                    for tau in [2*k, 4*k, 8*k, 16*k]
                    if tau <= n
                ]
                for algo in algos:
                    evaluate(dataset, delta, algo, k, ofile, shuffle_seed=shuffle_seed)


def streaming_experiments():
    ofile = "results.hdf5"
    ks = [32]
    deltas = [0.01]
    all_datasets = ["diabetes", "athlete", "census1990", "hmda"]
    #all_datasets = datasets.datasets()
    for dataset, delta, k in itertools.product(all_datasets, deltas, ks):
        for shuffle_seed in [1,2,3,4,5,6,7,8,9,10,11,12]:
            n, dim = datasets.dataset_size(dataset)
            algos = [
                streaming.BeraEtAlStreamingFairKCenter(
                    k, epsilon, cplex_path, seed=shuffle_seed)
                for epsilon in [0.5, 0.1, 0.05, 0.01]
            ] + [
                streaming.StreamingCoresetFairKCenter(
                    k, k*tau, cplex_path, seed=shuffle_seed)
                for tau in [128, 256, 512]
                #for tau in [2, 4, 8, 32, 128, 512]
                if tau <= n
            ]
            for algo in algos:
                evaluate(dataset, delta, algo, k, ofile, shuffle_seed=shuffle_seed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    signal.signal(signal.SIGALRM, timeout_handler)

    if len(sys.argv) == 2:
        cplex_path = sys.argv[1]
    else:
        cplex_path = None

    #exhaustive()
    #delta_influence()
    #mr_experiments()
    streaming_experiments()

