"""
The new algorithm gives higher priority to nodes based on a weighted combination of inverse travel time and node isolation,
while dynamically adjusting the importance of truck clustering over time.
"""


import numpy as np


# {The new algorithm gives higher priority to nodes based on a weighted combination of inverse travel time and node isolation, while dynamically adjusting the importance of truck clustering over time.}

def select_next_node(current, destination, trucks, unvisited):
    if not unvisited:
        return None

    best_score = float('-inf')
    next_destination = None

    current_position_array = np.array(current)

    # Calculate dynamic weight based on time
    total_nodes = len(unvisited)
    dynamic_weight = 1 - (total_nodes / (total_nodes + 1))

    for idx, node in enumerate(unvisited):
        node_array = np.array(node)
        travel_time = np.linalg.norm(node_array - current_position_array)

        isolation_score = np.sum([np.linalg.norm(node_array - np.array(truck)) for truck in trucks])

        score = (1 / (travel_time + 1)) + dynamic_weight * isolation_score

        if score > best_score:
            best_score = score
            next_destination = idx

    return next_destination