from assignment import weighted_fair_assignment, fair_assignment, freq_distributor
import time
from kcenter import greedy_minimum_maximum, assign_original_points
from sklearn.metrics import pairwise_distances
import pyspark
import logging
import pulp
import numpy as np


class MRCoresetFairKCenter(object):
    def __init__(self, k, tau, master, cplex_path=None, subroutine_name="freq_distributor", seed=42):
        self.k = k
        self.tau = tau
        self.seed = seed
        self.master = master
        self.subroutine_name = subroutine_name
        self.solver_cmd = pulp.CPLEX_CMD(
            path=cplex_path, msg=False) if cplex_path is not None else pulp.COIN_CMD(msg=False)

    def name(self):
        return "mr-coreset-fair-k-center"

    def time(self):
        return self.elapsed

    def attrs(self):
        return {
            "tau": self.tau,
            "seed": self.seed,
            "subroutine": self.subroutine_name
        }

    def additional_metrics(self):
        import assess
        return {
            "time_coreset_s": self.time_coreset_s,
            "time_assignment_s": self.time_assignment_s,
            "coreset_size": self.coreset_size
        }

    def fit_predict(self, X, colors, fairness_constraints):
        from collections import namedtuple
        start = time.time()

        tau = self.tau
        seed = self.seed
        ncolors = np.max(colors) + 1

        MRData = namedtuple("MRData", ["coreset_points",
                            "point_ids", "colors", "proxy", "coreset_weights"])

        def inner(arg):
            offset, (pts, colors) = arg
            point_ids, proxy = greedy_minimum_maximum(pts, tau, seed)
            coreset_points = pts[point_ids]

            point_ids += offset
            coreset_weights = np.zeros(
                (point_ids.shape[0], ncolors), dtype=np.int64)

            for color, proxy_idx in zip(colors, proxy):
                coreset_weights[proxy_idx, color] += 1

            return offset, MRData(coreset_points, point_ids, colors, proxy, coreset_weights)

        def parallel_assign(args):
            offset, (mrdata, assignmnent) = args
            print("offset", offset)
            print("mrdata", mrdata)

        # Build the coreset
        sc = pyspark.SparkContext(self.master)
        p = sc.defaultParallelism
        print("parallelism", p)
        splitdata = np.array_split(X, p, axis=0)
        splitcolor = np.array_split(colors, p, axis=0)
        X = []
        off = 0
        for dat, col in zip(splitdata, splitcolor):
            assert dat.shape[0] == col.shape[0]
            X.append((off, (dat, col)))
            off += dat.shape[0]

        Xp = sc.parallelize(X, p).repartition(p)
        coreset_p = Xp.map(inner).cache()
        coreset = coreset_p.collect()

        coreset_ids = []
        coreset_points = []
        coreset_weights = []
        coreset_proxy = []
        breaks = []
        offsets = []
        for offset, mrdata in sorted(coreset): # Sort by offset
            offsets.append(offset)
            coreset_ids.append(mrdata.point_ids)
            coreset_points.append(mrdata.coreset_points)
            coreset_weights.append(mrdata.coreset_weights)
            coreset_proxy.extend(mrdata.proxy + np.sum(breaks))
            breaks.append(mrdata.coreset_points.shape[0])
        coreset_ids = np.hstack(coreset_ids)
        coreset_proxy = np.hstack(coreset_proxy)
        coreset_points = np.vstack(coreset_points)
        coreset_weights = np.vstack(coreset_weights)
        self.time_coreset_s = time.time() - start
        self.coreset_size = np.flatnonzero(coreset_weights).shape[0]

        # Step 2. Find the greedy centers in the coreset
        centers, assignment = greedy_minimum_maximum(coreset_points, self.k)
        costs = pairwise_distances(coreset_points, coreset_points[centers])

        # Step 3. Find a fair assignment with the centers in the coreset
        subroutine = freq_distributor if self.subroutine_name == "freq_distributor" else weighted_fair_assignment
        coreset_centers, coreset_assignment = subroutine(
            centers, costs, coreset_weights, fairness_constraints, self.solver_cmd)

        # Step 4. Assign the input points to the centers found before
        centers, assignment = assign_original_points(
            self.k, colors, coreset_proxy, coreset_weights,
            coreset_ids, coreset_centers, coreset_assignment)
        self.assignment = assignment
        self.centers = centers
        self.time_assignment_s = time.time() - start - self.time_coreset_s

        end = time.time()
        self.elapsed = end - start

        return assignment
        


if __name__ == "__main__":
    import assess
    import datasets
    import kcenter

    logging.basicConfig(level=logging.INFO)

    cplex_path = "/home/matteo/opt/cplex/cplex/bin/x86-64_linux/cplex"

    master = "local[4]"
    k = 32
    delta = 0.01
    dataset = "4area"
    data, colors, fairness_constraints = datasets.load(
        dataset, 0, delta)

    tau = int(k*2)
    logging.info("Tau is %d", tau)
    algo = MRCoresetFairKCenter(k, tau, master, cplex_path)

    assignment = algo.fit_predict(data, colors, fairness_constraints)
    centers = algo.centers
    print("radius", assess.radius(data, centers, assignment))
    print("cluster sizes", assess.cluster_sizes(centers, assignment))
    print("violation", assess.additive_violations(
        k, colors, assignment, fairness_constraints))
    print(algo.attrs())
    print(algo.additional_metrics())
    print("time", algo.time())
