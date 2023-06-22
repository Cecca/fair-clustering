import sys
import numpy as np
import pandas as pd
import pulp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pulp import *
from pulp.apis import COIN_CMD
from numba import njit
from numba.typed import List
import datasets
import logging


def eucl(a, b):
    return np.sqrt(np.linalg.norm(a - b, axis=1))


def plot_clustering(data, centers, assignment, filename="clustering.png"):
    unique, counts = np.unique(assignment, return_counts=True)
    cluster = unique[np.argsort(-counts)]
    plt.figure(figsize=(10, 10))
    if data.shape[1] > 2:
        logging.info("projecting to 2 dimensions")
        data = PCA(n_components=2).fit_transform(data)

    for c in cluster:
        cdata = data[assignment == c]
        plt.scatter(cdata[:,0], cdata[:,1], s=10)

    plt.scatter(data[centers,0], data[centers,1], s=200, marker="x", c="black")

    plt.tight_layout()
    plt.savefig(filename)


def greedy_minimum_maximum(data, k, return_assignment=True, seed=123):
    np.random.seed(seed)
    centers = [np.random.choice(data.shape[0])]
    distances = eucl(data, data[centers[-1]])
    while len(centers) < k:
        farthest = np.argmax(distances)
        centers.append(farthest)
        distances = np.minimum(
            eucl(data, data[centers[-1]]),
            distances
        ) 
    
    centers = np.array(centers)

    if return_assignment:
        assignment = np.array([
            np.argmin(eucl(data[x], data[centers]))
            for x in range(data.shape[0])
        ])
        assert np.all(assignment[centers] == np.arange(k))
        return centers, assignment
    else:
        return centers


def cluster_radii(data, centers, assignment):
    return np.array([
        np.max(eucl(data[assignment == c], data[centers[c]]))
        for c in range(centers.shape[0])
        if np.any(assignment == c) # it might be that some cluster has even 
                                   # its center assigned to some other
    ])


def build_coreset(data, tau, colors, seed=42):
    point_ids, proxy = greedy_minimum_maximum(data, tau, seed=seed)
    # plot_clustering(data, point_ids, proxy, filename="coreset.png")
    ncolors = np.max(colors) + 1
    coreset_points = data[point_ids]
    coreset_weights = np.zeros((coreset_points.shape[0], ncolors), dtype=np.int64)
    for color, proxy_idx in zip(colors, proxy):
        coreset_weights[proxy_idx,color] += 1
    # cradius = cluster_radii(data, point_ids, proxy)
    # logging.info("coreset radius", np.max(cradius))
    return point_ids, coreset_points, proxy, coreset_weights


def do_fair_assignment(R, costs, weights, fairness_constraints):
    n, k = costs.shape
    ncolors = weights.shape[1]
    prob = LpProblem()
    vars = {}
    for x in range(n):
        for c in range(k):
            if costs[x,c] <= R:
                for color in range(ncolors):
                    if weights[x, color] > 0:
                        vars[x,c,color] = LpVariable(f"x_{x}_{c}_{color}", 0, cat="Continuous")
    
    # All the weight should be assigned
    for x in range(n):
        for color in range(ncolors):
            prob += (
                lpSum([vars[x,c,color] for c in range(k) if (x,c,color) in vars]) == np.sum(weights[x,color])
            )

    # The fairness constraints should be respected
    for c in range(k):
        cluster_size = lpSum([vars[x,c,color] for x in range(n) for color in range(ncolors) if (x,c,color) in vars])
        for color in range(ncolors):
            beta, alpha = fairness_constraints[color]
            color_cluster_size = lpSum([vars[x,c,color] for x in range(n) if (x,c,color) in vars])
            prob += (
                beta * cluster_size <= color_cluster_size
            )
            prob += (
                color_cluster_size <= alpha * cluster_size
            )

    prob.solve(COIN_CMD(mip=True, msg=False))
    if LpStatus[prob.status] == "Optimal":
        wassignment = {} #np.zeros((n,k,ncolors))
        for x in range(n):
            for c in range(k):
                for color in range(ncolors):
                    if (x,c,color) in vars:
                        v = vars[x,c,color]
                        if v.value() > 0:
                            wassignment[x,c,color] = v.value()
        return wassignment
    else:
        return None


def round_assignment(n, k, ncolors, weighted_assignment, weights):
    assignment = {}
    # Dictionaries for residual variables
    vars = {}
    weights = weights.copy()
    cluster_sizes = np.zeros(k)
    colored_cluster_sizes = np.zeros((k, ncolors))
    for (x, c, color) in weighted_assignment:
        z = weighted_assignment[x,c,color]
        floor = np.floor(z)
        assignment[x,c,color] = int(floor)
        weights[x, color] -= floor
        cluster_sizes[c] += z - floor
        colored_cluster_sizes[c,color] += z - floor
        if z != floor:
            vars[x,c,color] = LpVariable(f"z_{x}_{c}_{color}", 0, 1)

    blacklist = set()
    blacklist_colored = set()

    logging.info("variables to round: %d", len(vars))
    while len(vars) > 0:
        logging.info("There are still %d variables to round", len(vars))
        lp = LpProblem()
        for x in range(n):
            for color in range(ncolors):
                lp += (
                    lpSum([vars[x,c,color] for c in range(k) if (x,c,color) in vars]) == weights[x, color]
                )
        for c in range(k):
            if c in blacklist:
                continue
            csize = cluster_sizes[c]
            zsum = lpSum([
                vars[x,c,color] 
                for x in range(n) 
                for color in range(ncolors) 
                if (x,c,color) in vars
            ])
            lp += (zsum >= np.floor(csize))
            lp += zsum <= np.ceil(csize)
        for c in range(k):
            for color in range(ncolors):
                if (c,color) in blacklist_colored:
                    continue
                ccsize = colored_cluster_sizes[c,color]
                zsum = lpSum([
                    vars[x,c,color] 
                    for x in range(n) 
                    if (x,c,color) in vars
                ])
                lp += (zsum >= np.floor(ccsize))
                lp += zsum <= np.ceil(ccsize)

        lp.solve(COIN_CMD(mip=False, msg=False))
        assert LpStatus[lp.status] == "Optimal"
        blacklist_counts = {}
        blacklist_counts_colored = {}
        for (x,c,color) in vars.copy():
            if vars[x,c,color].value() == 0:
                del vars[x,c,color]
            elif vars[x,c,color].value() == 1:
                assignment[x,c,color] += 1
                cluster_sizes[c] -= 1
                colored_cluster_sizes[c,color] -= 1
                del vars[x,c,color]
            else:
                # Count how many point may go in a cluster
                blacklist_counts[c] += 1
                blacklist_counts_colored[c,color] += 1
        for c in blacklist_counts:
            if blacklist_counts[c] <= 3:
                blacklist.insert(c)
        for c,color in blacklist_counts_colored:
            if blacklist_counts_colored[c] <= 3:
                blacklist_colored.insert(c)

    return assignment


def _sum_cluster_weights(k, ncolors, assignment):
    s = np.zeros(( k, ncolors ))
    for (x, c, col) in assignment:
        s[c, col] += assignment[x,c,col]
    return s


def fair_assignment(k, coreset, weights, fairness_contraints):
    n = coreset.shape[0]
    ncolors = weights.shape[1]
    centers = greedy_minimum_maximum(coreset, k, return_assignment=False)
    costs = np.zeros((coreset.shape[0], k))
    for c in range(k):
        costs[:,c] = eucl(coreset, coreset[centers[c]])

    allcosts = np.sort(np.unique(costs))

    def binary_search():
        low, high = 0, allcosts.shape[0] 
        while low <= high:
            mid = low + (high - low) // 2
            R = allcosts[mid]
            logging.info("R %f", R)
            assignment = do_fair_assignment(R, costs, weights, fairness_contraints)
            logging.info("  clustering {}".format("feasible" if assignment is not None else "unfeasible"))
            if assignment is not None:
                cluster_weights = _sum_cluster_weights(k, ncolors, assignment)#.astype(np.int32)
                total_weight = np.sum(cluster_weights)
                logging.info("  cluster weights (total %f)\n%s", total_weight, cluster_weights)
            if low == high:
                return assignment
            if assignment is None:
                low = mid + 1
            else:
                if low == mid:
                    return assignment
                else:
                    high = mid

    wassignment = binary_search()
    assignment = round_assignment(n, k, len(fairness_contraints), wassignment, weights)
    return centers, assignment




def assign_original_points(colors, proxy, coreset_ids, coreset_centers, coreset_assignment, self_centers=False):
    logging.info("Assigning original points")
    centers = coreset_ids[coreset_centers]
    k = coreset_centers.shape[0]

    def assignment_to_budget(assignment):
        budget = np.zeros((k, colors.max() + 1), dtype=np.int32)
        for (p, c, color) in assignment:
            budget[c,color] += assignment[p,c,color]
        return budget

    # proxy_sizes = np.zeros((proxy.max() + 1, colors.max() + 1), dtype=np.int32)
    # for (color, p) in zip(colors, proxy):
    #     proxy_sizes[p,color] += 1
    # assert np.sum(proxy_sizes) == colors.shape[0]
    # print("proxy sizes")

    assignment = np.ones(colors.shape[0], dtype=np.int64) * 99999999
    for x, (color, p) in enumerate(zip(colors, proxy)):
        if self_centers and x in centers:
            assignment[x] = [i for i in range(k) if centers[i] == x][0]
        else:
            # look for the first cluster with budget for that color
            for c in range(k):
                if (p,c,color) in coreset_assignment:
                    if coreset_assignment[p,c,color] > 0:
                        candidates = [i for i in range(k) if centers[i] == coreset_ids[coreset_centers[c]]]
                        assert len(candidates) == 1
                        assignment[x] = candidates[0]
                        coreset_assignment[p,c,color] -= 1
                        break
    assert assignment.max() <= k, "there are some unassigned points!"
    return centers, assignment


@njit
def evaluate_fairness(k, colors, assignment, fairness_constraints):
    ncolors = np.max(colors) + 1
    counts = np.zeros((k, ncolors), dtype=np.int32)
    # Count the cluster size for each color
    for c, color in zip(assignment, colors):
        counts[c,color] += 1
    print("cluster sizes\n", counts)
    additive_violations = np.zeros((k, ncolors), dtype=np.int32)
    for c in range(k):
        cluster = counts[c]
        size = np.sum(cluster)
        for color in range(ncolors):
            beta, alpha = fairness_constraints[color]
            low, high = np.floor(beta * size), np.ceil(alpha * size)
            csize = counts[c,color]
            if csize < low:
                additive_violations[c,color] = csize - low
            elif csize > high:
                additive_violations[c,color] = csize - high
    return additive_violations
    

def main():
    k = 64
    dataset = "creditcard"
    data, colors, color_proportion = datasets.load(dataset, 0)

    delta = 0.1

    fairness_constraints = List([
        (p * (1-delta), p / (1-delta))
        for p in color_proportion
    ])

    # Greedy
    greedy_centers, greedy_assignment = greedy_minimum_maximum(data, k)
    plot_clustering(datasets.load_pca2(dataset), greedy_centers, greedy_assignment, filename="greedy.png")
    greedy_radius = cluster_radii(data, greedy_centers, greedy_assignment)
    logging.info("greedy radius %s", greedy_radius)
    logging.info("max greedy radius %s", np.max( greedy_radius ))
    greedy_violations = evaluate_fairness(k, colors, greedy_assignment, fairness_constraints)
    logging.info(greedy_violations)

    # Fair
    coreset_ids, coreset, proxy, weights = build_coreset(data, k*10, colors)
    logging.info("total weight %f", np.sum(weights))

    coreset_centers, coreset_assignment = fair_assignment(k, coreset, weights, fairness_constraints)
    centers, assignment = assign_original_points(colors, proxy, coreset_ids, coreset_centers, coreset_assignment, self_centers=True)
    fair_radius = cluster_radii(data, centers, assignment)
    logging.info("fair radius %s", fair_radius)
    logging.info("max fair radius %s", np.max( fair_radius ))
    plot_clustering(datasets.load_pca2(dataset), centers, assignment)
    violations = evaluate_fairness(k, colors, assignment, fairness_constraints)
    logging.info(violations)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
