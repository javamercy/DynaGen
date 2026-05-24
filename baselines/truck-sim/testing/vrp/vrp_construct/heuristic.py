import numpy as np


def select_next_node(current_node, truck_nodes, destination_node, unvisited_nodes, distance_matrix):
    if len(unvisited_nodes) == 0:
        return destination_node

    min_distance = np.inf
    next_node = None

    for node in unvisited_nodes:
        if distance_matrix[current_node][node] < min_distance:
            min_distance = distance_matrix[current_node][node]
            next_node = node

    return next_node
