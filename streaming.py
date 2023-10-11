from fairkcenter import streaming_build_coreset, build_assignment, greedy_minimum_maximum
from assignment import weighted_fair_assignment, fair_assignment, freq_distributor
import time
from sklearn.metrics import pairwise_distances
import logging
import pulp
import numpy as np


class StreamingCoresetFairKCenter(object):
    def __init__(self, k, tau, cplex_path=None, subroutine_name="freq_distributor", seed=42):
        self.k = k
        self.tau = tau
        self.guess_step = 2
        self.seed = seed
        self.subroutine_name = subroutine_name
        self.solver_cmd = pulp.CPLEX_CMD(
            path=cplex_path, msg=False) if cplex_path is not None else pulp.COIN_CMD(msg=False)

    def name(self):
        return "streaming-coreset-fair-k-center"

    def time(self):
        return self.elapsed

    def attrs(self):
        return {
            "tau": self.tau,
            "seed": self.seed,
            "subroutine": self.subroutine_name,
        }

    def additional_metrics(self):
        import assess
        return {
            "time_coreset_s": self.time_coreset_s,
            "time_assignment_s": self.time_assignment_s,
            "coreset_size": self.coreset_size,
            "streaming_memory_bytes": self.total_size_bytes
        }

    def fit_predict(self, X, colors, fairness_constraints):
        tau = self.tau

        # Build the coreset
        start = time.time()
        coreset_ids, coreset_points, coreset_proxy, coreset_weights, total_size_bytes = streaming_build_coreset(
            X, self.guess_step, self.k, tau, colors)
        self.time_coreset_s = time.time() - start
        self.total_size_bytes = total_size_bytes
        assert coreset_weights.sum() == X.shape[0]

        self.coreset_size = np.flatnonzero(coreset_weights).shape[0]

        # Step 2. Find the greedy centers in the coreset
        centers, assignment = greedy_minimum_maximum(coreset_points, self.k)
        costs = pairwise_distances(coreset_points, coreset_points[centers])
        assert len(centers) == self.k

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


class BeraEtAlStreamingFairKCenter(StreamingCoresetFairKCenter):
    def __init__(self, k, epsilon, cplex_path=None, subroutine_name="bera-et-al", seed=42):
        self.k = k
        self.tau = k
        self.epsilon = epsilon
        self.guess_step = 1 + epsilon
        self.seed = seed
        self.subroutine_name = subroutine_name
        self.solver_cmd = pulp.CPLEX_CMD(
            path=cplex_path, msg=False) if cplex_path is not None else pulp.COIN_CMD(msg=False)

    def name(self):
        return "bera-streaming-fair-k-center"

    def attrs(self):
        return {
            "epsilon": self.epsilon,
            "seed": self.seed,
            "subroutine": self.subroutine_name,
        }


if __name__ == "__main__":
    import pandas as pd
    import datasets
    import assess
    import os

    logging.basicConfig(level=logging.INFO)

    cplex_path = os.path.expanduser('~/opt/cplex/cplex/bin/x86-64_linux/cplex')

    k = 32
    delta = 0.01
    all_datasets = ["hmda", "census1990_age"]

    df = []
    for dataset in all_datasets:
        for shuffle_seed in range(1):
            data, colors, fairness_constraints = datasets.load(
                dataset, 0, delta, shuffle_seed=shuffle_seed)
            for tau in [1]:
                algo = StreamingCoresetFairKCenter(k, k*tau, cplex_path, subroutine_name="freq_distributor")
                assignment = algo.fit_predict(data, colors, fairness_constraints)
                res = {
                    "algorithm": algo.name(),
                    "param": tau,
                    "time": algo.time(),
                    "radius": assess.radius(data, algo.centers, assignment)
                }
                res.update(algo.additional_metrics())
                df.append(res)

            # for epsilon in [0.5, 0.1, 0.05, 0.01]:
            #     algo = BeraEtAlStreamingFairKCenter(k, epsilon, cplex_path)
            #     assignment = algo.fit_predict(data, colors, fairness_constraints)
            #     res = {
            #         "algorithm": algo.name(),
            #         "param": epsilon,
            #         "time": algo.time(),
            #         "radius": assess.radius(data, algo.centers, assignment)
            #     }
            #     res.update(algo.additional_metrics())
            #     df.append(res)

    df = pd.DataFrame(df)
    print(df)
    print(df.groupby(['algorithm', 'param']).mean())

