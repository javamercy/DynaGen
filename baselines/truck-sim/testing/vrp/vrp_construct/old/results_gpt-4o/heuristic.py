import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Calculate distance from the current node to all unvisited nodes
    current_distances = distance_matrix[current_node, unvisited_nodes]

    # Calculate distance from each unvisited node to the destination node
    destination_distances = distance_matrix[unvisited_nodes, destination_node]

    # Calculate inter-node distances among unvisited nodes
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    inter_node_avg_distances = sub_matrix.mean(axis=1)

    # Calculate the detour score: proximity benefit among unvisited nodes with respect to the destination
    detour_scores = current_distances + 0.5 * inter_node_avg_distances - 0.8 * destination_distances

    # Select the unvisited node with the minimum detour score
    next_node_index = np.argmin(detour_scores)
    next_node = unvisited_nodes[next_node_index]

    return next_node
