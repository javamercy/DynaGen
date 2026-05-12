import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    edge_factor = 5
    percentage = 0.6
    unused_edges = np.where(edge_n_used == 0)
    num_edges = len(unused_edges[0])
    num_edges_to_update = int(num_edges * percentage)

    for _ in range(num_edges_to_update):
        idx = np.random.randint(num_edges)
        edge1 = unused_edges[0][idx]
        edge2 = unused_edges[1][idx]

        if edge1 in local_opt_tour and edge2 in local_opt_tour:
            updated_edge_distance[edge1][edge2] += edge_factor
            updated_edge_distance[edge2][edge1] += edge_factor

    return updated_edge_distance
