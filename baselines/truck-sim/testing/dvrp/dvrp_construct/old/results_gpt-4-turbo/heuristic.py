import numpy as np


def select_next_node(current, destination, trucks, unvisited):
    """
    Algorithm prioritizes neighboring distance reduction based on modified nearest neighbor strategy enhanced by zoning effects of other truck locations.
    """
    # Calculate distances from current position to each unvisited node
    distances = np.linalg.norm(unvisited - current, axis=1)

    # Add a dynamic influence zone factor which depends on truck density near each unvisited node
    zone_influence = np.zeros(len(unvisited))
    scale_factor = 1.5  # Adjusts influence of zone factor

    for idx, node in enumerate(unvisited):
        truck_density = 0
        for truck in trucks:
            distance_to_truck = np.linalg.norm(node - truck)
            truck_density += np.exp(-distance_to_truck)  # Using exponential decay to account for truck proximity
        zone_influence[idx] = truck_density

    # Normalize truck density influence across all unvisited nodes
    if np.max(zone_influence) > 0:
        zone_influence /= np.max(zone_influence)

    # Adjust original distances by normalized truck density influence
    adjusted_scores = distances + (scale_factor * zone_influence)

    # Select the unvisited node with the minimum adjusted distance score as next destination
    next_destination = np.argmin(adjusted_scores)

    return next_destination
