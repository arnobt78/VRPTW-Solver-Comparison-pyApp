from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from pyvrp import ProblemData


def plot_my_solution(
    solution: any,
    data: ProblemData,
    ax: Optional[plt.Axes] = None,
    dataset: str = "c101",
    algo: str = "HGS",
):
    """
    Plots the given solution.

    Parameters
    ----------
    solution
        Solution to plot.
    data
        Data instance underlying the solution.
    plot_clients
        Whether to plot clients as dots. Default False, which plots only the
        solution's routes.
    ax
        Axes object to draw the plot on. One will be created if not provided.
    """
    if not ax:
        _, ax = plt.subplots()

    dim = data.num_clients + 1
    x_coords = np.array([data.client(client).x for client in range(dim)])
    y_coords = np.array([data.client(client).y for client in range(dim)])

    # This is the depot
    kwargs = dict(c="tab:red", marker="*", zorder=3, s=500)
    ax.scatter(x_coords[0], y_coords[0], label="Depot", **kwargs)

    for idx, route in enumerate(solution["routes"], 1):
        x = x_coords[route]
        y = y_coords[route]

        # Coordinates of clients served by this route.
        ax.scatter(x, y, label=f"Route {idx}", zorder=3, s=75)
        ax.plot(x, y)

        # Edges from and to the depot, very thinly dashed.
        kwargs = dict(ls=(0, (5, 15)), linewidth=0.25, color="grey")
        ax.plot([x_coords[0], x[0]], [y_coords[0], y[0]], **kwargs)
        ax.plot([x[-1], x_coords[0]], [y[-1], y_coords[0]], **kwargs)

    ax.grid(color="grey", linestyle="solid", linewidth=0.2)

    ax.set_title(f"Solution - {dataset} - {algo} - {solution['cost']:.1f}")
    ax.set_aspect("equal", "datalim")
    ax.legend(frameon=False, ncol=2)
