from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np
from icecream import ic
import sys

sys.path.append("..")
import kcenter

cplex_path = "/home/matteo/opt/cplex/cplex/bin/x86-64_linux/cplex"


def bezier(p1, p2, amount=0.4, ax=None, color="black", lw=1):
    verts = [
        tuple(p1),  # P0
        (p1[0] + (p2[0] - p1[0]) * amount, p1[1]),  # z1
        (p2[0], p2[1] - (p2[1] - p1[1]) * amount),  # z2
        tuple(p2),  # P2
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
    ]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor="none", lw=lw, edgecolor=color, zorder=-1)
    if ax is None:
        ax = plt.gca()
    ax.add_patch(patch)


def plot_points(
    fname,
    data,
    colors,
    color_map=["#0072b2", "#d55e00"],
    symb_map=["▲", "■"],
    marker_map=["^", "s"],
    linestyle_map=dict(),
    show_weights=False,
    color_points=True,
    highlight=None,
    assignment=None,
    show_lines=True,
    hull=None,
    xlims=None,
    ylims=None,
    figsize=(4, 5),
):
    plt.figure(figsize=figsize)
    for col in [0, 1]:
        pcolor = color_map[col]
        if show_weights:
            pcolor = "white"
        if not color_points:
            pcolor = "gray"
        plt.scatter(
            data[colors == col, 0],
            data[colors == col, 1],
            c=pcolor,
            marker=marker_map[col],
        )
    # for i in range(data.shape[0]):
    #     plt.text(data[i, 0], data[i, 1], str(i))

    if highlight is not None:
        for idx in highlight:
            col = colors[idx]
            pcolor = color_map[col]
            if not color_points:
                pcolor = "gray"
            plt.scatter(
                data[idx, 0],
                data[idx, 1],
                c=pcolor,
                s=100,
                marker=marker_map[col],
                edgecolors="black",
                linewidths=2,
            )

    if assignment is not None and len(assignment.shape) == 1:
        if not show_weights and show_lines:
            for point_id, center_id in enumerate(assignment):
                p = data[point_id]
                c = data[center_id]
                plt.plot(
                    [p[0], c[0]],
                    [p[1], c[1]],
                    c="gray",
                    linestyle=linestyle_map.get(center_id, "solid"),
                    linewidth=0.5,
                    zorder=-1,
                )
        for c in highlight:
            for col in [0, 1]:
                assigned_of_color = np.sum(
                    np.logical_and(assignment == c, colors == col)
                )
                cpoint = data[c]
                yoff = 0.15
                xoff = -0.25
                if col == 1:
                    yoff *= -1
                plt.text(
                    cpoint[0] + xoff,
                    cpoint[1] + yoff,
                    f"{symb_map[col]} {assigned_of_color}",
                    ha="right",
                    va="center",
                    c=color_map[col] if show_weights else "white",
                    alpha=1.0 if show_weights else 0.0,
                    bbox=dict(
                        edgecolor="white",
                        facecolor="white",
                        alpha=0.8 if show_weights else 0.0,
                    ),
                )
    elif assignment is not None and len(assignment.shape) == 3:
        max_assignment = np.max(assignment)
        for assignee, point_assignment in enumerate(assignment):
            for center, colors_assignments in enumerate(point_assignment):
                for col, col_amount in enumerate(colors_assignments):
                    p = data[assignee]
                    c = data[center]
                    if col == 0:
                        bezier(
                            p,
                            c,
                            lw=4 * col_amount / max_assignment,
                            color=color_map[col],
                        )
                    else:
                        bezier(
                            c,
                            p,
                            lw=4 * col_amount / max_assignment,
                            color=color_map[col],
                        )

    if hull is not None:
        import shapely

        colors = ["red", "green"]
        for c in highlight:
            pdata = data[assignment == c]
            pts = shapely.multipoints(pdata)
            chull_x, chull_y = shapely.convex_hull(pts).buffer(hull).exterior.xy
            plt.plot(chull_x, chull_y, c="black", lw=0.8)

    if xlims is not None:
        plt.xlim(xlims)
    if ylims is not None:
        plt.ylim(ylims)
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(fname, bbox_inches="tight", transparent=False, pad_inches=0, dpi=150)


def simulation(n=20, seed=12345):
    gen = np.random.default_rng(seed)

    g1 = gen.normal((0, 0), size=(n, 2))
    g2 = gen.normal((0.0, 2.0), size=(n, 2))

    data = np.concatenate((g1, g2))
    colors = np.array([0] * n + [1] * n).astype(np.int64)
    delta = 0.0
    fairness_constraints = [(p * (1 - delta), p / (1 - delta)) for p in [0.5, 0.5]]

    plot_points("imgs/input-data.png", data, colors)

    clusterer = kcenter.CoresetFairKCenter(2, 10, cplex_path, subroutine_name="wfd")
    assignment = clusterer.fit_predict(data, colors, fairness_constraints)
    plot_points(
        "imgs/coreset.png",
        data,
        colors,
        highlight=clusterer.coreset_points_ids,
        assignment=clusterer.coreset_points_ids[clusterer.proxy],
        show_weights=False,
        hull=0.2,
        show_lines=False,
    )
    xlims = plt.xlim()
    ylims = plt.ylim()
    plot_points(
        "imgs/coreset-weights.png",
        data,
        colors,
        highlight=clusterer.coreset_points_ids,
        assignment=clusterer.coreset_points_ids[clusterer.proxy],
        show_weights=True,
        xlims=xlims,
        ylims=ylims,
    )

    plot_points(
        "imgs/coreset-unfair-clustering.png",
        data[clusterer.coreset_points_ids],
        colors[clusterer.coreset_points_ids],
        color_points=False,
        marker_map=["o", "o"],
        highlight=clusterer.unfair_coreset_centers,
        assignment=clusterer.unfair_coreset_centers[
            clusterer.unfair_coreset_assignment
        ],
        xlims=xlims,
        ylims=ylims,
        hull=0.2,
        show_lines=True,
    )

    plot_points(
        "imgs/coreset-fair-clustering.png",
        data[clusterer.coreset_points_ids],
        colors[clusterer.coreset_points_ids],
        highlight=clusterer.fair_coreset_centers,
        assignment=clusterer.fair_coreset_assignment,
        color_points=False,
        marker_map=["o", "o"],
        xlims=xlims,
        ylims=ylims,
    )

    plot_points(
        "imgs/coreset-final-fair-clustering.png",
        data,
        colors,
        highlight=clusterer.centers,
        assignment=clusterer.centers[assignment],
        xlims=xlims,
        ylims=ylims,
        hull=0.2,
        show_lines=True,
    )


def example(
    n=20,
    seed=1234,
    color_map=["#0072b2", "#d55e00"],
    symb_map=["▲", "■"],
    marker_map=["^", "s"],
):
    from scipy.spatial import ConvexHull

    gen = np.random.default_rng(seed)

    g1 = gen.normal((0, 0), size=(n, 2))
    g2 = gen.normal((8.0, 0.0), size=(n, 2))

    data = np.concatenate((g1, g2))
    colors = np.array([0] * n + [1] * n).astype(np.int64)
    delta = 0.0
    fairness_constraints = [(p * (1 - delta), p / (1 - delta)) for p in [0.5, 0.5]]

    # unfair = kcenter.UnfairKCenter(2)
    # unfair_assignment = unfair.fit_predict(data, colors, fairness_constraints)
    unfair_centers = np.array([6, 37])
    unfair_assignment = np.array([0] * n + [1] * n).astype(np.int64)

    plot_points(
        "imgs/example-unfair.png",
        data,
        colors,
        highlight=unfair_centers,
        assignment=unfair_centers[unfair_assignment],
        show_lines=False,
        figsize=(5, 3),
        hull=0.4,
    )

    fair_centers = np.array([6, 37])
    fair = kcenter.BeraEtAlKCenter(2, cplex_path, init_centers=fair_centers)
    fair_assignment = fair.fit_predict(data, colors, fairness_constraints)
    plot_points(
        "imgs/example-fair.png",
        data,
        colors,
        highlight=fair.centers,
        assignment=fair.centers[fair_assignment],
        show_lines=False,
        figsize=(5, 3),
        hull=0.4,
    )


simulation(n=20)
# example(seed=1234)
