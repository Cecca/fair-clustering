import sys
import numpy as np
import pandas as pd
import pulp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
from pulp import *
from pulp.apis import COIN_CMD
import datasets


def eucl(a, b):
    return np.sqrt(np.linalg.norm(a - b, axis=1))


def plot_clustering(data, centers, assignment):
    unique, counts = np.unique(assignment, return_counts=True)
    cluster = unique[np.argsort(-counts)]
    plt.figure(figsize=(10, 10))
    if data.shape[1] > 2:
        print("projecting to 2 dimensions", file=sys.stderr)
        data = PCA(n_components=2).fit_transform(data)

    for c in cluster:
        cdata = data[assignment == c]
        plt.scatter(cdata[:,0], cdata[:,1], s=10)

    plt.scatter(data[centers,0], data[centers,1], s=200, marker="x", c="black")

    plt.tight_layout()
    plt.savefig("clustering.png")


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
    ncolors = np.max(colors) + 1
    coreset_points = data[point_ids]
    coreset_weights = np.zeros((coreset_points.shape[0], ncolors), dtype=np.int64)
    for color, proxy_idx in zip(colors, proxy):
        coreset_weights[proxy_idx,color] += 1
    # cradius = cluster_radii(data, point_ids, proxy)
    # print("coreset radius", np.max(cradius))
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
                        vars[x,c,color] = LpVariable(f"x_{x}_{c}_{color}", 0)
    
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

    prob.solve(COIN_CMD(mip=False, msg=False))
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

    while len(vars) > 0:
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


def fair_assignment(k, coreset, weights, fairness_contraints):
    n = coreset.shape[0]
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
            print("R", R)
            assignment = do_fair_assignment(R, costs, weights, fairness_contraints)
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
    return centers, round_assignment(n, k, len(fairness_contraints), wassignment, weights)


def assign_original_points(colors, proxy, coreset_ids, coreset_centers, coreset_assignment):
    centers = coreset_ids[coreset_centers]
    k = coreset_centers.shape[0]
    assignment = np.zeros(colors.shape[0], dtype=np.int64)
    for x, (color, p) in enumerate(zip(colors, proxy)):
        # look for the first cluster with budget for that color
        for c in range(k):
            if (p,c,color) in coreset_assignment and coreset_assignment[p,c,color] > 0:
                assignment[x] = [i for i in range(k) if centers[i] == coreset_ids[coreset_centers[c]]][0]
                coreset_assignment[p,c,color] -= 1
    return centers, assignment


def main():
    k = 16
    dataset = "diabetes"
    data, colors, color_proportion = datasets.load(dataset, 0)

    greedy_centers, greedy_assignment = greedy_minimum_maximum(data, k)
    greedy_radius = cluster_radii(data, greedy_centers, greedy_assignment)
    print("greedy radius", greedy_radius)
    print("max greedy radius", np.max( greedy_radius ))

    delta = 0.01

    fairness_contraints = [
        (p * (1-delta), p / (1-delta))
        for p in color_proportion
    ]

    coreset_ids, coreset, proxy, weights = build_coreset(data, k*40, colors)

    coreset_centers, coreset_assignment = fair_assignment(k, coreset, weights, fairness_contraints)
    centers, assignment = assign_original_points(colors, proxy, coreset_ids, coreset_centers, coreset_assignment)
    fair_radius = cluster_radii(data, centers, assignment)
    print("fair radius", fair_radius)
    print("max fair radius", np.max( fair_radius ))
    plot_clustering(datasets.load_umap(dataset), centers, assignment)


if __name__ == "__main__":
    main()
