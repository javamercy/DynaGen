import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    dist_active_to_customers = np.linalg.norm(available_customers - current_position, axis=1)
    dist_customers_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)

    # Find the index of the active truck in truck_positions
    # current_position is exactly one of the truck positions (by design of the simulation)
    active_index = np.where(np.all(truck_positions == current_position, axis=1))[0][0]

    # Distances from all trucks to all customers: shape (n_cust, n_trucks)
    all_truck_dists = np.linalg.norm(
        available_customers[:, np.newaxis, :] - truck_positions[np.newaxis, :, :], axis=2
    )

    # Minimum distance to each customer among trucks other than active
    other_dists = np.copy(all_truck_dists)
    other_dists[:, active_index] = np.inf
    min_other_dists = np.min(other_dists, axis=1)

    # Is the active truck the closest (or tied) for each customer?
    active_is_closest = all_truck_dists[:, active_index] <= min_other_dists

    if np.any(active_is_closest):
        candidate_indices = np.where(active_is_closest)[0]
        # Among candidates, pick the one farthest from depot
        best_idx = candidate_indices[np.argmax(dist_customers_to_depot[candidate_indices])]
    else:
        # Active is not closest to any customer, pick the nearest customer
        best_idx = np.argmin(dist_active_to_customers)

    return int(best_idx)