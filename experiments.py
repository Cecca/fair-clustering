import sys
import mapreduce
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


def evaluate(dataset, delta, algo, k, ofile):
    if results.already_run(dataset, algo.name(), k, delta, algo.attrs()):
        return
    logging.info(
        f"Running {algo.name()} with {algo.attrs()} on {dataset} for k={k}")
    data, colors, fairness_constraints = datasets.load(
        dataset, 0, delta)

    try:
        signal.alarm(TIMEOUT_SECS)
        assignment = algo.fit_predict(data, colors, fairness_constraints)
        signal.alarm(0)  # Cancel the alarm
        centers = algo.centers

        results.save_result(ofile, centers, assignment, dataset,
                            algo.name(), k, delta, algo.attrs(),
                            algo.time(), algo.additional_metrics())
    except TimeoutException:
        results.save_timeout(dataset, algo.name(), k,
                             delta, algo.attrs(), TIMEOUT_SECS)

def warmup(cplex_path):
    logging.basicConfig(level=logging.WARNING)
    k = 3
    delta = 0.0
    dataset = "random_dbg"
    data, colors, fairness_constraints = datasets.load(
        dataset, 0, delta)
    n, dims = datasets.dataset_size(dataset)
    tau = 10
    algo = kcenter.CoresetFairKCenter(k, tau, cplex_path, seed=2)
    algo.fit_predict(data, colors, fairness_constraints)
    logging.basicConfig(level=logging.INFO)


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
    ks = [32]
    # ks = [2, 4, 8, 16, 32]
    deltas = [0.01]
    all_datasets = datasets.datasets()
    for dataset, delta, k in itertools.product(all_datasets, deltas, ks):
        n, dim = datasets.dataset_size(dataset)
        algos = [
            # kcenter.Dummy(k),
            # kcenter.UnfairKCenter(k),
            # kcenter.BeraEtAlKCenter(k, cplex_path),
            # KFC(k, cplex_path)
        ] + [
            kcenter.CoresetFairKCenter(
                k, tau, cplex_path, seed=seed)
            for tau in [2*k, 8*k, 32*k, 64*k, 128*k, 256*k]
            for seed in [1,2,3,4,5,6,7]
            if tau <= n
        ]
        for algo in algos:
            evaluate(dataset, delta, algo, k, ofile)


def mr_experiments():
    ofile = "results.hdf5"
    ks = [32]
    deltas = [0.01]
    all_datasets = ["census1990", "hmda", "athlete"]
    for dataset, delta, k in itertools.product(all_datasets, deltas, ks):
        n, dim = datasets.dataset_size(dataset)
        for threads in [2, 4, 8, 16, 32]:
            master = f"local[{threads}]"
            algos = [
                mapreduce.BeraEtAlMRFairKCenter(
                    k, master, cplex_path, seed=seed)
                for seed in [1]
            ] + [
                mapreduce.MRCoresetFairKCenter(
                    k, tau, master, cplex_path, seed=seed)
                for tau in [2*k, 8*k, 32*k, 64*k, 128*k]
                for seed in [1]
                if tau <= n
            ]
            for algo in algos:
                evaluate(dataset, delta, algo, k, ofile)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    signal.signal(signal.SIGALRM, timeout_handler)

    if len(sys.argv) == 2:
        cplex_path = sys.argv[1]
        warmup(cplex_path)
    else:
        cplex_path = None

    exhaustive()
    delta_influence()
    mr_experiments()

