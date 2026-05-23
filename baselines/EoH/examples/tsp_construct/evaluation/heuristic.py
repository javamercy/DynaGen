# example heuristic
# replace it with your own heuristic designed by EoH
import numpy as np


def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray,
                     distance_matrix: np.ndarray) -> int:
    """Select the next node to visit in a TSP greedy construction.

    Args:
        current_node: ID of the current node
        destination_node: ID of the destination (return) node
        unvisited_nodes: array of unvisited node IDs
        distance_matrix: pairwise distance matrix between all nodes
    Returns:
        next_node: ID of the next node to visit
    """
    total_nodes = distance_matrix.shape[0]
    visited = total_nodes - len(unvisited_nodes)
    beta = visited / total_nodes  # fraction of visited nodes
    dist_curr = distance_matrix[current_node, unvisited_nodes]
    dist_dest = distance_matrix[destination_node, unvisited_nodes]
    # Multiplicative combination: product of dist_curr and (1 + beta * dist_dest)
    scores = dist_curr * (1 + beta * dist_dest)
    best_idx = np.argmin(scores)
    return unvisited_nodes[best_idx]
