import os
import pickle

import numpy as np


def generate_data(prob, problem_size: list[int], n_test: int, n_thread: int = 1):
    result = prob.get_data(50, 10, n_thread)
    dataset = [r[:-2] for r in result]
    results = [r[-2:] for r in result]

    final = [d + (r[0],) for d, r in zip(dataset, results)]
    filename = os.path.join("./training_data_1", f"instances.pkl")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(final, f)

    final = [d + (r[1],) for d, r in zip(dataset, results)]
    filename = os.path.join("./training_data_3", f"instances.pkl")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(final, f)

    for size in problem_size:
        result = prob.get_data(size, n_test, n_thread)
        dataset = [r[:-2] for r in result]
        results = [r[-2:] for r in result]

        final = [d + (r[0],) for d, r in zip(dataset, results)]
        filename = os.path.join("testing_data_1", f"instance_data_{size}.pkl")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(final, f)

        final = [d + (r[1],) for d, r in zip(dataset, results)]
        filename = os.path.join("testing_data_3", f"instance_data_{size}.pkl")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(final, f)


def plot_solution(coordinates, routes, ax, plot_locations: bool = False):
    num_depots = 1
    size = len(coordinates)
    x_coords = np.array([c[0] for c in coordinates])
    y_coords = np.array([c[1] for c in coordinates])

    # plot the depot
    kwargs = dict(c="tab:red", marker="*", zorder=3, s=500)
    ax.scatter(
        x_coords[: num_depots],
        y_coords[: num_depots],
        label="Depot",
        **kwargs,
    )

    # plot the routes
    in_solution = np.zeros(size, dtype=bool)
    for idx, route in enumerate(routes):
        in_solution[route] = True

        x = x_coords[route][1:-1]
        y = y_coords[route][1:-1]

        # Coordinates of clients served by this route.
        if len(route) == 1 or plot_locations:
            ax.scatter(x, y, label=f"Route {idx}", zorder=3, s=75)
        ax.plot(x, y)

        # Thin edges from and to the depot. The edge from the depot to the
        # first client is given an arrow head to indicate route direction. We
        # don't do this for the edge returning to the depot because that adds a
        # lot of clutter at the depot.
        start_depot = route[0]
        end_depot = route[0]

        if len(x) > 0 and len(y) > 0:
            kwargs = dict(linewidth=0.25, color="grey")
            ax.plot(
                [x[-1], x_coords[end_depot]],
                [y[-1], y_coords[end_depot]],
                linewidth=0.25,
                color="grey",
            )
            ax.annotate(
                "",
                xy=(x[0], y[0]),
                xytext=(x_coords[start_depot], y_coords[start_depot]),
                arrowprops=dict(arrowstyle="-|>", **kwargs),
                zorder=1,
            )
    
    # Plot points that have not been visited
    unvisited = np.flatnonzero(~in_solution[num_depots :])
    x = x_coords[num_depots + unvisited]
    y = y_coords[num_depots + unvisited]
    ax.scatter(x, y, label="Unvisited", zorder=3, s=75, c="grey")

    # Add gridlines
    ax.grid(color="grey", linestyle="solid", linewidth=0.2)
