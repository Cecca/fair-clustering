import numpy as np
from sklearn.metrics import pairwise_distances
import logging
import time

import datasets
from assignment import weighted_fair_assignment, fair_assignment


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
        np.random.seed(self.seed)

        centers = [np.random.choice(X.shape[0])]
        distances = pairwise_distances(X, X[centers[-1]].reshape(1, -1))
        while len(centers) < self.k:
            farthest = np.argmax(distances)
            centers.append(farthest)
            distances = np.minimum(
                pairwise_distances(X, X[centers[-1]].reshape(1, -1)),
                distances
            )
        self.centers = np.array(centers)

        self.assignment = np.array([
            np.argmin(pairwise_distances(X[centers], X[x].reshape(1, -1)))
            for x in range(X.shape[0])
        ])
        assert np.all(self.assignment[centers] == np.arange(self.k))
        end = time.time()
        self.elapsed = end - start
        return self.assignment


class BeraEtAlKCenter(object):
    def __init__(self, k, seed=123):
        self.k = k
        self.seed = seed

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
            centers, costs, colors, fairness_constraints)
        self.centers = centers
        self.assignment = assignment

        end = time.time()
        self.elapsed = end - start
        return self.assignment


def greedy_minimum_maximum(data, k, return_assignment=True, seed=123):
    np.random.seed(seed)
    centers = [np.random.choice(data.shape[0])]
    distances = pairwise_distances(data, data[centers[-1]].reshape(1, -1))
    while len(centers) < k:
        farthest = np.argmax(distances)
        centers.append(farthest)
        distances = np.minimum(
            pairwise_distances(data, data[centers[-1]].reshape(1, -1)),
            distances
        )

    centers = np.array(centers)

    if return_assignment:
        assignment = np.array([
            np.argmin(pairwise_distances(
                data[centers], data[x].reshape(1, -1)))
            for x in range(data.shape[0])
        ])
        assert np.all(assignment[centers] == np.arange(k))
        return centers, assignment
    else:
        return centers


class CoresetFairKCenter(object):
    def __init__(self, k, tau, integer_programming=False, seed=42):
        self.k = k
        self.tau = tau
        self.seed = seed
        self.integer_programming = integer_programming

    def name(self):
        return "coreset-fair-k-center"

    def time(self):
        return self.elapsed

    def attrs(self):
        return {
            "tau": self.tau,
            "seed": self.seed,
            "integer_programming": self.integer_programming
        }

    def additional_metrics(self):
        return {
            "time_coreset_s": self.time_coreset_s,
            "time_assignment_s": self.time_assignment_s
        }

    def fit_predict(self, X, colors, fairness_constraints):
        # Step 1. Build the coreset
        start = time.time()
        coreset_ids, coreset, proxy, weights = self.build_coreset(
            X, self.tau, colors)
        self.time_coreset_s = time.time() - start

        # Step 2. Find the greedy centers in the coreset
        centers = greedy_minimum_maximum(
            coreset, self.k, return_assignment=False)
        costs = pairwise_distances(coreset, coreset[centers])

        # Step 3. Find a fair assignment with the centers in the coreset
        coreset_centers, coreset_assignment = weighted_fair_assignment(
            centers, costs, weights, fairness_constraints)

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
                        assert len(candidates) == 1
                        assignment[x] = candidates[0]
                        coreset_assignment[p, c, color] -= 1
                        break
        assert assignment.max() <= k, "there are some unassigned points!"
        return centers, assignment


if __name__ == "__main__":
    import viz
    import assess
    logging.basicConfig(level=logging.INFO)

    k = 8
    delta = 0.1
    dataset = "reuter_50_50"
    # dataset = "creditcard"
    data, colors, fairness_constraints = datasets.load(
        dataset, 0, delta, prefix=10000)

    # Fair
    tau = k*10
    algo = CoresetFairKCenter(k, tau)
    assignment = algo.fit_predict(data, colors, fairness_constraints)
    centers = algo.centers
    print(assignment)
    print("radius", assess.radius(data, centers, assignment))
    viz.plot_clustering(data, centers, assignment, "clustering.png")
