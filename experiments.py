import kcenter
import results
import datasets
import itertools
import logging


# FIXME: add timeout


def evaluate(dataset, delta, algo):
    if results.already_run(dataset, algo.name(), k, delta, algo.attrs()):
        return
    logging.info(f"Running {algo.name()} on {dataset} for k={k}")
    data, colors, fairness_constraints = datasets.load(
        dataset, 0, delta)

    assignment = algo.fit_predict(data, colors, fairness_constraints)
    centers = algo.centers

    results.save_result(ofile, centers, assignment, dataset,
                        algo.name(), k, delta, algo.attrs(),
                        algo.time(), algo.additional_metrics())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ofile = "results.hdf5"
    ks = [2, 4, 8, 16, 32]
    deltas = [0]  # , 0.1, 0.2]
    all_datasets = ["creditcard"]  # , "diabetes", "adult"]
    for dataset, delta, k in itertools.product(all_datasets, deltas, ks):
        algos = [
            kcenter.UnfairKCenter(k),
            kcenter.BeraEtAlKCenter(k),
        ] + [
            kcenter.CoresetFairKCenter(k, tau, integer_programming=False)
            for tau in [2*k, 8*k]
        ]
        for algo in algos:
            evaluate(dataset, delta, algo)
