"""Visualization utilities"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import logging


def plot_clustering(data, centers, assignment, size=None, filename="clustering.png"):
    unique, counts = np.unique(assignment, return_counts=True)
    cluster = unique[np.argsort(-counts)]
    plt.figure(figsize=(10, 10))
    if data.shape[1] > 2:
        logging.info("projecting to 2 dimensions")
        data = PCA(n_components=2).fit_transform(data)

    for c in cluster:
        cdata = data[assignment == c]
        plt.scatter(cdata[:, 0], cdata[:, 1])

    plt.scatter(data[centers, 0], data[centers, 1],
                s=200, marker="x", c="black")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_dataset(name, filename="dataset.png"):
    import datasets
    data = datasets.load_pca2(name)
    if data is None:
        return
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1])
    plt.tight_layout()
    plt.savefig(filename)


if __name__ == "__main__":
    import datasets
    import sys
    import results
    import os

    # for dataset in datasets.datasets():
    #     plot_dataset(dataset, f"data/{dataset}.png")

    if len(sys.argv) == 3:
        resfile = sys.argv[1]
        imgdir = sys.argv[2]
        keys = results.list_keys(resfile)
        for key in keys:
            img_path = os.path.join(imgdir, key)
            os.makedirs(img_path, exist_ok=True)
            img_path = os.path.join(img_path, "clustering.png")
            if not os.path.isfile(img_path):
                print("Plotting image for", key)
                centers, assignment, dataset = results.read_key(resfile, key)
                data = datasets.load_pca2(dataset)
                plot_clustering(data, centers, assignment, filename=img_path)
