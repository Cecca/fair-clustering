from fairkcenter import parallel_build_coreset, build_assignment, greedy_minimum_maximum
from assignment import weighted_fair_assignment, fair_assignment, freq_distributor
import time
from sklearn.metrics import pairwise_distances
import logging
import pulp
import numpy as np
from collections import namedtuple


MRData = namedtuple("MRData", ["coreset_points",
                    "point_ids", "colors", "proxy", "coreset_weights"])


# class Reducer(object):
#     def __init__(self, tau, ncolors, seed):
#         self.tau = tau
#         self.ncolors = ncolors
#         self.seed = seed
#
#     def __call__(self, arg):
#         import cProfile
#         import pstats
#
#         offset, (pts, colors) = arg
#         greedy_minimum_maximum(pts[:10,:], 2, self.seed) # precompile on the current process
#
#         pr = cProfile.Profile()
#         pr.enable()
#         point_ids, proxy = greedy_minimum_maximum(pts, self.tau, self.seed)
#         coreset_points = pts[point_ids]
#
#         point_ids += offset
#         coreset_weights = np.zeros(
#             (point_ids.shape[0], self.ncolors), dtype=np.int64)
#
#         for color, proxy_idx in zip(colors, proxy):
#             coreset_weights[proxy_idx, color] += 1
#
#         ret = MRData(coreset_points, point_ids, colors, proxy, coreset_weights)
#
#         pr.disable()
#         id = multiprocessing.current_process()
#         with open(f"profile-{id}.txt", "w") as fp:
#             sortby = pstats.SortKey.CUMULATIVE
#             ps = pstats.Stats(pr, stream=fp).sort_stats(sortby)
#             ps.print_stats()
#
#         return offset, ret
#

class MRCoresetFairKCenter(object):
    def __init__(self, k, tau, parallelism, cplex_path=None, subroutine_name="freq_distributor", seed=42):
        self.k = k
        self.tau = tau
        self.seed = seed
        self.subroutine_name = subroutine_name
        self.solver_cmd = pulp.CPLEX_CMD(
            path=cplex_path, msg=False) if cplex_path is not None else pulp.COIN_CMD(msg=False)
        self.parallelism = parallelism

    def name(self):
        return "mr-coreset-fair-k-center"

    def time(self):
        return self.elapsed

    def attrs(self):
        return {
            "tau": self.tau,
            "seed": self.seed,
            "subroutine": self.subroutine_name,
            "parallelism": self.parallelism
        }

    def additional_metrics(self):
        import assess
        return {
            "time_coreset_s": self.time_coreset_s,
            "time_assignment_s": self.time_assignment_s,
            "coreset_size": self.coreset_size
        }

    def fit_predict(self, X, colors, fairness_constraints):
        tau = self.tau

        # Build the coreset
        start = time.time()
        coreset_ids, coreset_points, coreset_proxy, coreset_weights = parallel_build_coreset(
            self.parallelism, X, tau, colors)
        self.time_coreset_s = time.time() - start
        print("Coreset built in", self.time_coreset_s, "seconds")
        print(coreset_ids)
        print("max coreset proxy",max(coreset_proxy))

        self.coreset_size = np.flatnonzero(coreset_weights).shape[0]

        # Step 2. Find the greedy centers in the coreset
        centers, assignment = greedy_minimum_maximum(coreset_points, self.k)
        costs = pairwise_distances(coreset_points, coreset_points[centers])

        # Step 3. Find a fair assignment with the centers in the coreset
        subroutine = freq_distributor if self.subroutine_name == "freq_distributor" else weighted_fair_assignment
        coreset_centers, coreset_assignment = subroutine(
            centers, costs, coreset_weights, fairness_constraints, self.solver_cmd)

        # Step 4. Assign the input points to the centers found before
        centers, assignment = build_assignment(
            colors, coreset_proxy, coreset_ids, coreset_centers, coreset_assignment)
        self.assignment = assignment
        self.centers = centers
        self.time_assignment_s = time.time() - start - self.time_coreset_s

        end = time.time()
        self.elapsed = end - start

        return assignment
        

class BeraEtAlMRFairKCenter(MRCoresetFairKCenter):
    def __init__(self, k, parallelism, cplex_path=None, seed=42):
        super().__init__(k, k, parallelism, cplex_path, subroutine_name="bera-et-al", seed=seed)

    def name(self):
        return "bera-mr-fair-k-center"


if __name__ == "__main__":
    import pandas as pd
    import datasets
    import assess

    logging.basicConfig(level=logging.INFO)

    cplex_path = "/home/ceccarello/opt/cplex/cplex/bin/x86-64_linux/cplex"

    k = 32
    delta = 0.01
    dataset = "census1990"
    data, colors, fairness_constraints = datasets.load(
        dataset, 0, delta)

    tau = int(k*32)
    logging.info("Tau is %d", tau)
    df = []
    for parallelism in [2,4,8,16]:
        algo = MRCoresetFairKCenter(k, tau, parallelism, cplex_path)

        assignment = algo.fit_predict(data, colors, fairness_constraints)
        df.append({
            "parallelism": parallelism,
            "time": algo.time(),
            "coreset_time": algo.additional_metrics()["time_coreset_s"],
            "radius": assess.radius(data, algo.centers, assignment)
        })
    df = pd.DataFrame(df)
    print(df)
