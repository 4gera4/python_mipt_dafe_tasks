from typing import Any

import matplotlib.pyplot as plt
import numpy as np


class ShapeMismatchError(Exception):
    pass


def change_color_violin(violin_parts):
    for vp in violin_parts:
        plt.setp(vp["bodies"], facecolor="#C71585", edgecolor="#FF1493")

        line_keys = set(vp.keys()) - {"bodies"}
        for key in line_keys:
            plt.setp(vp[key], edgecolor="#C71585")


def visualize_diagrams(
    abscissa: np.ndarray,
    ordinates: np.ndarray,
    diagram_type: Any,
) -> None:
    if abscissa.shape != ordinates.shape:
        raise ShapeMismatchError

    valid_types = ["hist", "violin", "box"]

    if diagram_type not in valid_types:
        raise ValueError

    space = 0.2

    figure = plt.figure(figsize=(8, 8))
    grid = plt.GridSpec(4, 4, wspace=space, hspace=space)
    axis_scatter = figure.add_subplot(grid[:-1, 1:])
    axis_hist_vert = figure.add_subplot(
        grid[:-1, 0],
        sharey=axis_scatter,
    )
    axis_hist_hor = figure.add_subplot(
        grid[-1, 1:],
        sharex=axis_scatter,
    )
    axis_scatter.scatter(abscissa, ordinates, color="#FF1493", alpha=0.5)

    if diagram_type == "hist":
        axis_hist_hor.hist(
            abscissa,
            bins=50,
            color="#FF1493",
            density=True,
            alpha=0.5,
        )
        axis_hist_vert.hist(
            ordinates,
            bins=50,
            color="#FF1493",
            orientation="horizontal",
            density=True,
            alpha=0.5,
        )

        axis_hist_hor.invert_yaxis()
    if diagram_type == "box":
        axis_hist_hor.boxplot(
            abscissa,
            vert=False,
            patch_artist=True,
            boxprops=dict(facecolor="#C71585"),
            medianprops=dict(color="#FF1493"),
        )
        axis_hist_vert.boxplot(
            ordinates,
            patch_artist=True,
            boxprops=dict(facecolor="#C71585"),
            medianprops=dict(color="#FF1493"),
        )
        axis_hist_hor.invert_yaxis()
        axis_hist_hor.set_yticks([])
        axis_hist_vert.set_xticks([])

    if diagram_type == "violin":
        violin_parts = np.empty(2, dtype=object)
        violin_parts[0] = axis_hist_hor.violinplot(
            abscissa,
            vert=False,
            showmedians=True,
        )
        violin_parts[1] = axis_hist_vert.violinplot(
            ordinates,
            showmedians=True,
        )
        axis_hist_hor.invert_yaxis()
        axis_hist_hor.set_yticks([])
        axis_hist_vert.set_xticks([])
        change_color_violin(violin_parts)

    axis_hist_vert.invert_xaxis()


if __name__ == "__main__":
    mean = [2, 3]
    cov = [[1, 1], [1, 2]]
    space = 0.2

    abscissa, ordinates = np.random.multivariate_normal(mean, cov, size=1000).T

    visualize_diagrams(abscissa, ordinates, "box")
    plt.show()
