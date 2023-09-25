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
            "coreset_size": self.coreset_size
        }

    def fit_predict(self, X, colors, fairness_constraints):
        tau = self.tau

        # Build the coreset
        start = time.time()
        coreset_ids, coreset_points, coreset_proxy, coreset_weights = streaming_build_coreset(
            X, 2, self.k, tau, colors)
        self.time_coreset_s = time.time() - start
        print("Coreset built in", self.time_coreset_s, "seconds")
        print(coreset_ids)
        print(coreset_weights)
        assert coreset_weights.sum() == X.shape[0]

        self.coreset_size = np.flatnonzero(coreset_weights).shape[0]

        # Step 2. Find the greedy centers in the coreset
        centers, assignment = greedy_minimum_maximum(coreset_points, self.k)
        costs = pairwise_distances(coreset_points, coreset_points[centers])
        assert len(centers) == self.k
        print(centers)
        print(costs)

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


if __name__ == "__main__":
    import pandas as pd
    import datasets
    import assess
    import os

    logging.basicConfig(level=logging.INFO)

    cplex_path = os.path.expanduser('~/opt/cplex/cplex/bin/x86-64_linux/cplex')

    k = 2
    delta = 0.01
    dataset = "adult"
    data, colors, fairness_constraints = datasets.load(
        dataset, 0, delta)

    tau = int(k*8)
    logging.info("Tau is %d", tau)
    df = []
    algo = StreamingCoresetFairKCenter(k, tau, cplex_path, subroutine_name="freq_distributor")

    assignment = algo.fit_predict(data, colors, fairness_constraints)
    df.append({
        "time": algo.time(),
        "coreset_time": algo.additional_metrics()["time_coreset_s"],
        "radius": assess.radius(data, algo.centers, assignment)
    })
    df = pd.DataFrame(df)
    print(df)

