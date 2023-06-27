"""Visualization utilities"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import logging


def plot_clustering(data, centers, assignment, filename="clustering.png"):
    unique, counts = np.unique(assignment, return_counts=True)
    cluster = unique[np.argsort(-counts)]
    plt.figure(figsize=(10, 10))
    if data.shape[1] > 2:
        logging.info("projecting to 2 dimensions")
        data = PCA(n_components=2).fit_transform(data)

    for c in cluster:
        cdata = data[assignment == c]
        plt.scatter(cdata[:, 0], cdata[:, 1], s=10)

    plt.scatter(data[centers, 0], data[centers, 1],
                s=200, marker="x", c="black")

    plt.tight_layout()
    plt.savefig(filename)


def plot_dataset(name, filename="dataset.png"):
    import datasets
    data = datasets.load_pca2(name)
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1])
    plt.tight_layout()
    plt.savefig(filename)


if __name__ == "__main__":
    import datasets
    for dataset in datasets.datasets():
        plot_dataset(dataset, f"data/{dataset}.png")
