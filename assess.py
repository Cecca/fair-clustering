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


