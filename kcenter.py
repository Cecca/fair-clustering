import numpy as np
from sklearn.metrics import pairwise_distances
import logging
import time
import pulp

import datasets
from assignment import weighted_fair_assignment, fair_assignment, freq_distributor


class UnfairKCenter(object):
    def __init__(self, k, seed=123):
        self.k = k
        self.seed = seed

    def attrs(self):
        return {}

    def additional_metrics(self):
        return {}

    def name(self):
        return "unfair-k-center"

    def time(self):
        return self.elapsed

    def fit_predict(self, X, _colors_unused, _fairness_constraints_unused):
        start = time.time()
        centers, assignment = greedy_minimum_maximum(
            X, self.k, return_assignment=True, seed=self.seed)
        self.centers = centers
        self.assignment = assignment

        end = time.time()
        self.elapsed = end - start
        return self.assignment


class BeraEtAlKCenter(object):
    def __init__(self, k, cplex_path=None, seed=123):
        self.k = k
        self.seed = seed
        self.solver_cmd = pulp.CPLEX_CMD(
            path=cplex_path, msg=0) if cplex_path is not None else pulp.COIN_CMD(msg=False)
        logging.info("solver is %s", self.solver_cmd)

    def attrs(self):
        return {}

    def additional_metrics(self):
        return {}

    def name(self):
        return "bera-et-al-k-center"

    def time(self):
        return self.elapsed

    def fit_predict(self, X, colors, fairness_constraints):
        start = time.time()

        # Step 1. Find centers
        centers = greedy_minimum_maximum(
            X, self.k, return_assignment=False, seed=self.seed)
        costs = pairwise_distances(X, X[centers])

        # Step 2. Find assignment
        centers, assignment = fair_assignment(
            centers, costs, colors, fairness_constraints, solver=self.solver_cmd)
        self.centers = centers
        self.assignment = assignment

        end = time.time()
        self.elapsed = end - start
        return self.assignment


def greedy_minimum_maximum(data, k, return_assignment=True, seed=123, p=None):
    np.random.seed(seed)
    first_center = np.random.choice(data.shape[0], p=p)
    centers = [first_center]
    distances = pairwise_distances(
        data, data[centers[-1]].reshape(1, -1))[:, 0]
    assignment = np.zeros(data.shape[0], np.int32)
    idx = 0
    while len(centers) < k:
        idx += 1
        farthest = np.argmax(distances)
        centers.append(farthest)
        distances_to_new_center = pairwise_distances(
            data, data[farthest].reshape(1, -1))[:, 0]
        # update the assignment if we found a closest center
        assignment[distances_to_new_center < distances] = idx
        # update the distances
        distances = np.minimum(
            distances_to_new_center,
            distances
        )

    centers = np.array(centers)

    if return_assignment:
        return centers, assignment
    else:
        return centers


class CoresetFairKCenter(object):
    def __init__(self, k, tau, cplex_path=None, subroutine_name="freq_distributor", seed=42):
        self.k = k
        self.tau = tau
        self.seed = seed
        self.subroutine_name = subroutine_name
        self.solver_cmd = pulp.CPLEX_CMD(
            path=cplex_path, msg=False) if cplex_path is not None else pulp.COIN_CMD(msg=False)
        logging.info("solver is %s", self.solver_cmd)

    def name(self):
        return "coreset-fair-k-center"

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
        coreset_radius = assess.radius(
            self.data, self.coreset_points_ids, self.proxy)
        logging.info("Coreset radius %f", coreset_radius)
        return {
            "time_coreset_s": self.time_coreset_s,
            "time_assignment_s": self.time_assignment_s,
            "coreset_radius": coreset_radius
        }

    def fit_predict(self, X, colors, fairness_constraints):
        assert self.tau <= X.shape[
            0], f"Tau larger than number of points {self.tau} > {X.shape[0]}"
        # Step 1. Build the coreset
        self.data = X
        start = time.time()
        coreset_ids, coreset, proxy, weights = self.build_coreset(
            X, self.tau, colors)
        self.time_coreset_s = time.time() - start

        # Step 2. Find the greedy centers in the coreset
        # selection_probabilities = np.sum(weights, axis=1).astype(np.float64)
        # selection_probabilities /= np.sum(selection_probabilities)
        centers, assignment = greedy_minimum_maximum(
            coreset, self.k, return_assignment=True, p=None)
        costs = pairwise_distances(coreset, coreset[centers])

        # viz.plot_clustering(datasets.load_pca2("4area")[coreset_ids], centers, assignment,
        #                     filename="coreset-clustering.png")

        # Step 3. Find a fair assignment with the centers in the coreset
        subroutine = freq_distributor if self.subroutine_name == "freq_distributor" else weighted_fair_assignment
        coreset_centers, coreset_assignment = subroutine(
            centers, costs, weights, fairness_constraints, self.solver_cmd)

        # Step 4. Assign the input points to the centers found before
        centers, assignment = self.assign_original_points(
            colors, proxy, coreset_ids, coreset_centers, coreset_assignment, self_centers=False)
        self.time_assignment_s = time.time() - start - self.time_coreset_s

        end = time.time()
        self.elapsed = end - start
        self.centers = centers
        self.assignment = assignment
        return assignment

    def build_coreset(self, data, tau, colors):
        point_ids, proxy = greedy_minimum_maximum(data, tau, seed=self.seed)
        self.coreset_points_ids = point_ids
        self.proxy = proxy
        ncolors = np.max(colors) + 1
        coreset_points = data[point_ids]
        coreset_weights = np.zeros(
            (coreset_points.shape[0], ncolors), dtype=np.int64)
        for color, proxy_idx in zip(colors, proxy):
            coreset_weights[proxy_idx, color] += 1
        return point_ids, coreset_points, proxy, coreset_weights

    def assign_original_points(self, colors, proxy, coreset_ids, coreset_centers, coreset_assignment, self_centers=False):
        logging.info("Assigning original points")
        centers = coreset_ids[coreset_centers]
        k = coreset_centers.shape[0]

        assignment = np.ones(colors.shape[0], dtype=np.int64) * 99999999
        for x, (color, p) in enumerate(zip(colors, proxy)):
            if self_centers and x in centers:
                assignment[x] = [i for i in range(k) if centers[i] == x][0]
            else:
                # look for the first cluster with budget for that color
                for c in range(k):
                    # if (p, c, color) in coreset_assignment:
                    if coreset_assignment[p, c, color] > 0:
                        candidates = [i for i in range(
                            k) if centers[i] == coreset_ids[coreset_centers[c]]]
                        # assert len(candidates) == 1, candidates
                        assignment[x] = candidates[0]
                        coreset_assignment[p, c, color] -= 1
                        break
        assert assignment.max() <= k, "there are some unassigned points!"
        return centers, assignment


if __name__ == "__main__":
    import cProfile
    import pstats
    import io
    from pstats import SortKey

    import viz
    import assess
    from baseline.adapter import KFC
    logging.basicConfig(level=logging.INFO)

    cplex_path = "/home/matteo/opt/cplex/cplex/bin/x86-64_linux/cplex"

    k = 64
    delta = 0.0
    dataset = "census1990"
    data, colors, fairness_constraints = datasets.load(
        dataset, 0, delta)
    n, dims = datasets.dataset_size(dataset)
    viz.plot_dataset(dataset, "dataset.png")

    # Fair
    tau = int(128)
    logging.info("Tau is %d", tau)
    algo = CoresetFairKCenter(k, tau, cplex_path, seed=2)
    # algo = KFC(k, cplex_path, seed=2)
    # algo = BeraEtAlKCenter(k, cplex_path, seed=2)
    print(f"{algo.name()} ==============")
    assignment = algo.fit_predict(data, colors, fairness_constraints)
    centers = algo.centers
    print("radius", assess.radius(data, centers, assignment))
    print("violation", assess.additive_violations(
        k, colors, assignment, fairness_constraints))
    print(algo.attrs())
    print(algo.additional_metrics())
    print("time", algo.time())
    viz.plot_clustering(data, centers, assignment, filename="clustering.png")
