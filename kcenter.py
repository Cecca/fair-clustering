import numpy as np
from numba import njit, prange
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


@njit
def eucl(a, b, sq_norm_a, sq_norm_b):
    return np.sqrt(sq_norm_a + sq_norm_b - 2*np.dot(a, b))


@njit(parallel=True)
def set_eucl(vec, data, sq_norm_vec, sq_norms, out):
    """Computes the Euclidean distance between a vector 
    and a set of vectors, in parallel"""
    # res = np.zeros(data.shape[0], np.float64)
    for i in prange(data.shape[0]):
        out[i] = eucl(vec, data[i], sq_norm_vec, sq_norms[i])
    # return res


def _test_set_eucl():
    data = np.random.standard_normal((100, 4))
    v = data[0]
    sq_norms = np.zeros(data.shape[0])
    for i in prange(data.shape[0]):
        sq_norms[i] = np.dot(data[i], data[i])
    expected = pairwise_distances(data, v.reshape(1, -1))[:, 0]
    actual = np.zeros(data.shape[0])
    set_eucl(v, data, sq_norms[0], sq_norms, actual)
    assert np.all(np.isclose(expected, actual))


@njit
def element_wise_minimum(a, b, out):
    for i in range(a.shape[0]):
        if a[i] < b[i]:
            out[i] = a[i]
        else:
            out[i] = b[i]


@njit()
def greedy_minimum_maximum(data, k, seed=123):
    np.random.seed(seed)
    sq_norms = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        sq_norms[i] = np.dot(data[i], data[i])
    first_center = np.random.choice(data.shape[0])
    centers = np.zeros(k, np.int32)
    centers[0] = first_center
    distances = np.zeros(data.shape[0])
    distances_to_new_center = np.zeros(data.shape[0])
    set_eucl(data[first_center], data,
             sq_norms[first_center], sq_norms, distances)
    assignment = np.zeros(data.shape[0], np.int32)
    for idx in range(1, k):
        farthest = np.argmax(distances)
        centers[idx] = farthest
        set_eucl(
            data[farthest], data, sq_norms[farthest], sq_norms, distances_to_new_center)
        # update the assignment if we found a closest center
        assignment[distances_to_new_center < distances] = idx
        # update the distances

        element_wise_minimum(distances_to_new_center, distances, distances)
    return centers, assignment


def assign_original_points(k, colors, proxy, weights, coreset_ids, coreset_centers, coreset_assignment):
    @njit
    def inner():
        centers = coreset_ids[coreset_centers]
        nproxies, k, ncolors = coreset_assignment.shape

        assignment = np.ones(colors.shape[0], dtype=np.int64) * 99999999

        for c in range(k):
            for p in range(nproxies):
                if np.any(coreset_assignment[p, c] > 0):
                    weight_to_distribute = coreset_assignment[p, c].copy()
                    proxied = np.nonzero(proxy == p)[0]
                    for x in proxied:
                        color = colors[x]
                        if weight_to_distribute[color] > 0 and assignment[x] > k:
                            assignment[x] = c
                            weight_to_distribute[color] -= 1
                    assert np.sum(weight_to_distribute) == 0

        return centers, assignment

    t_start = time.time()
    logging.info("Assigning original points")
    centers, assignment = inner()
    logging.info("Assignment of original points took %f seconds",
                 time.time() - t_start)
    assert assignment.max() <= k, "there are some unassigned points!"
    return centers, assignment


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
        centers, assignment = greedy_minimum_maximum(coreset, self.k)
        costs = pairwise_distances(coreset, coreset[centers])

        # Step 3. Find a fair assignment with the centers in the coreset
        subroutine = freq_distributor if self.subroutine_name == "freq_distributor" else weighted_fair_assignment
        coreset_centers, coreset_assignment = subroutine(
            centers, costs, weights, fairness_constraints, self.solver_cmd)

        # Step 4. Assign the input points to the centers found before
        centers, assignment = assign_original_points(
            self.k, colors, proxy, weights, coreset_ids, coreset_centers, coreset_assignment)
        self.time_assignment_s = time.time() - start - self.time_coreset_s

        end = time.time()
        self.elapsed = end - start
        self.centers = centers
        self.assignment = assignment
        return assignment

    def build_coreset(self, data, tau, colors):
        timer = time.time()
        point_ids, proxy = greedy_minimum_maximum(
            data, tau, seed=self.seed)
        logging.info("GMM in %f s", time.time() - timer)
        timer = time.time()
        self.coreset_points_ids = point_ids
        self.proxy = proxy
        ncolors = np.max(colors) + 1
        coreset_points = data[point_ids]
        coreset_weights = np.zeros(
            (coreset_points.shape[0], ncolors), dtype=np.int64)
        for color, proxy_idx in zip(colors, proxy):
            coreset_weights[proxy_idx, color] += 1
        logging.info("assignment in %f s", time.time() - timer)
        return point_ids, coreset_points, proxy, coreset_weights


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

    def warmup():
        logging.basicConfig(level=logging.WARNING)
        _test_set_eucl()
        k = 3
        delta = 0.0
        dataset = "random_dbg"
        data, colors, fairness_constraints = datasets.load(
            dataset, 0, delta)
        n, dims = datasets.dataset_size(dataset)
        tau = 10
        algo = CoresetFairKCenter(k, tau, cplex_path, seed=2)
        algo.fit_predict(data, colors, fairness_constraints)
        logging.basicConfig(level=logging.INFO)

    warmup()

    k = 8
    delta = 0.0
    dataset = "adult"
    data, colors, fairness_constraints = datasets.load(
        dataset, 0, delta)
    n, dims = datasets.dataset_size(dataset)
    # viz.plot_dataset(dataset, "dataset.png")

    # Fair
    tau = int(1024)
    logging.info("Tau is %d", tau)
    algo = CoresetFairKCenter(
        k, tau, cplex_path, seed=2, subroutine_name="freq_distributor")
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
    # viz.plot_clustering(data, centers, assignment, filename="clustering.png")
