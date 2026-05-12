import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    rankings = np.zeros(len(unvisited_nodes))
    for i, node in enumerate(unvisited_nodes):
        if node == destination_node:
            rankings[i] = 0
        else:
            distance_to_node = distance_matrix[current_node][node]
            euclidean_distance = np.linalg.norm(distance_matrix[node] - distance_matrix[destination_node])
            if distance_to_node == 0:
                rankings[i] = np.inf
            else:
                rankings[i] = euclidean_distance / distance_to_node

    next_node = unvisited_nodes[np.argmax(rankings)]

    return next_node
