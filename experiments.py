import kcenter
import results
import assess
import viz
import datasets
import itertools
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ofile = "results.hdf5"
    ks = [2,4,8,16,32,64]
    deltas = [0, 0.1, 0.2]
    all_datasets = ["creditcard", "diabetes", "adult"]
    for dataset, delta, k in itertools.product(all_datasets, deltas, ks):
        # Greedy
        algo = kcenter.UnfairKCenter(k)
        if not results.already_run(dataset, algo.name(), k, delta, {}):
            data, colors, fairness_constraints = datasets.load(dataset, 0, delta)

            assignment = algo.fit_predict(data)
            centers = algo.centers

            results.save_result(ofile, centers, assignment, dataset, algo.name(), k, delta, {}, algo.time(), {})

        # Fair coreset
        for tau in [2*k, 8*k]:
            print(dataset,delta,k,tau)
            algo = kcenter.CoresetFairKCenter(k, tau, integer_programming=True)
            if not results.already_run(dataset, algo.name(), k, delta, algo.attrs()):
                data, colors, fairness_constraints = datasets.load(dataset, 0, delta)

                assignment = algo.fit_predict(data, colors, fairness_constraints)
                centers = algo.centers

                results.save_result(ofile, centers, assignment, dataset, algo.name(), k, delta, algo.attrs(), algo.time(), algo.additional_metrics())
        

