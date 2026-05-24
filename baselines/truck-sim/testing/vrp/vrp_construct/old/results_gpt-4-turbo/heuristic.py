import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    if len(unvisited_nodes) == 0:
        return destination_node

    # Environmental Awareness Algorithm using inverse distance attractiveness and an attraction to the center of mass of all unvisited nodes
    center_of_mass = np.mean(unvisited_nodes)

    # Calculate distances to unvisited nodes
    distances = distance_matrix[current_node, unvisited_nodes]
    inverse_distances = 1 / distances

    # Calculate attractiveness based on how close nodes are to the center of mass
    center_attraction = np.abs(unvisited_nodes - center_of_mass)

    # Normalize scores
    normalized_inverse_distances = inverse_distances / np.max(inverse_distances)
    normalized_center_attraction = center_attraction / np.max(center_attraction)

    # Combine scores with a bias towards inverse distance
    scores = 0.7 * normalized_inverse_distances + 0.3 * (1 - normalized_center_attraction)

    # Select node with highest score
    next_node_index = np.argmax(scores)
    next_node = unvisited_nodes[next_node_index]

    return next_node
