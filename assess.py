"""
Functions to assess the quality of clusterings.
"""

import numpy as np
from sklearn.metrics.pairwise import paired_distances
from sklearn.metrics import DistanceMetric
import datasets
import results

def radius(data, centers, assignment, all=False):
    k = centers.shape[0]
    dists = paired_distances(data, data[centers[assignment]], metric="euclidean")

    if all:
        r = np.zeros(k)
        for c in range(k):
            cdists = dists[assignment == c]
            if len(cdists) > 0:
                r[c] = np.max(cdists)
        return r
    else:
        return np.max(dists)


def additive_violations(k, colors, assignment, fairness_constraints):
    ncolors = np.max(colors) + 1
    counts = np.zeros((k, ncolors), dtype=np.int32)
    # Count the cluster size for each color
    for c, color in zip(assignment, colors):
        counts[c,color] += 1
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
    return np.abs(additive_violations).max()


if __name__ == "__main__":
    f = "results.hdf5"
    dataset = "creditcard"
    algorithm = "fair-coreset"
    k = 64
    attrs = {
        "delta": 0.1,
        "tau": 640
    }
    data, colors, color_proportion = datasets.load(dataset, 0)
    centers, assignment = results.read_clustering(f, dataset, algorithm, k, attrs)
    r = radius(data, centers, assignment, all=True)
    print(r)


