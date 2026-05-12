import numpy as np
import random


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    penalty_factor = 0.5
    avail_nodes = np.copy(unvisited_nodes)
    avail_nodes = avail_nodes[avail_nodes != current_node]

    if len(avail_nodes) == 1:
        return avail_nodes.item()

    curr_dist = distance_matrix[current_node, avail_nodes]
    dest_dist = distance_matrix[destination_node, avail_nodes]

    scores = curr_dist - dest_dist + penalty_factor / (1 / np.abs(curr_dist))

    next_node = avail_nodes[np.argmin(scores)]

    return next_node
