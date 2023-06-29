import kcenter
import results
import datasets
import itertools
import logging
import signal

TIMEOUT_SECS = 30*60


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    logging.warning("Timed out!")
    raise TimeoutException()


def evaluate(dataset, delta, algo):
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    signal.signal(signal.SIGALRM, timeout_handler)

    ofile = "results.hdf5"
    ks = [2, 4, 8, 16, 32]
    deltas = [0]  # , 0.1, 0.2]
    all_datasets = datasets.datasets()
    for dataset, delta, k in itertools.product(all_datasets, deltas, ks):
        n, dim = datasets.dataset_size(dataset)
        algos = [
            kcenter.UnfairKCenter(k),
            kcenter.BeraEtAlKCenter(k),
        ] + [
            kcenter.CoresetFairKCenter(
                k, tau, seed=seed, integer_programming=False)
            for tau in [2*k, 8*k, 32*k, 64*k, 128*k, 256*k, 512*k, 1024*k, 2048*k]
            for seed in range(1, 3)
            if tau <= n
        ]
        for algo in algos:
            evaluate(dataset, delta, algo)
