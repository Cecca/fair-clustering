import kcenter
import results
import assess
import viz
import datasets
import itertools

if __name__ == "__main__":
    ofile = "results.hdf5"
    ks = [2,4,8,16,32,64]
    deltas = [0, 0.1, 0.2]
    all_datasets = ["creditcard", "diabetes", "adult"]
    for dataset, delta, k in itertools.product(all_datasets, deltas, ks):
        # Greedy
        algo = kcenter.UnfairKCenter(k)
        if results.already_run(dataset, algo.name(), k, delta, {}):
            continue
        data, colors, fairness_constraints = datasets.load(dataset, 0, delta)

        assignment = algo.fit_predict(data)
        centers = algo.centers

        results.save_result(ofile, centers, assignment, dataset, algo.name(), k, delta, {}, algo.time())
        

