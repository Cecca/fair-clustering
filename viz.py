"""Visualization utilities"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.decomposition import PCA
import logging


def plot_clustering(data, centers, assignment, r=None, size=None, filename="clustering.png"):
    unique, counts = np.unique(assignment, return_counts=True)
    cluster = unique[np.argsort(-counts)]
    plt.figure(figsize=(10, 10))
    if data.shape[1] > 2:
        logging.info("projecting to 2 dimensions")
        data = PCA(n_components=2).fit_transform(data)

    for c in cluster:
        cdata = data[assignment == c]
        plt.scatter(cdata[:, 0], cdata[:, 1], s=2)

    plt.scatter(data[centers, 0], data[centers, 1],
                s=200, marker="x", c="black")
    if r is not None:
        limits = plt.xlim(), plt.ylim()
        for c in range(centers.shape[0]):
            xy = tuple(data[centers[c]])
            print(xy)
            circ = Circle(xy, r, fill=False, clip_on=True)
            plt.gca().add_patch(circ)
        plt.xlim(limits[0])
        plt.ylim(limits[1])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_dataset(name, filename="dataset.png"):
    import datasets
    data, colors = datasets.load_pca2(name)
    plot_clustering(data, [], colors, filename=filename)
    # if data is None:
    #     return
    # plt.figure(figsize=(10, 10))
    # plt.scatter(data[:, 0], data[:, 1])
    # plt.tight_layout()
    # plt.savefig(filename)


if __name__ == "__main__":
    import datasets
    import sys
    import results
    import os

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
                data, colors = datasets.load_pca2(dataset)
                plot_clustering(data, centers, assignment, filename=img_path)
    else:
        for dataset in datasets.datasets():
            plot_dataset(dataset, f"data/{dataset}.png")
