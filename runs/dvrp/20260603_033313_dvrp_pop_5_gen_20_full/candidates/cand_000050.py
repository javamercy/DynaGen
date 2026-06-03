import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    ALPHA = 1.0

    dist_active_to_customers = np.linalg.norm(available_customers - current_position, axis=1)
    dist_customers_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)

    active_index = np.where(np.all(truck_positions == current_position, axis=1))[0][0]

    all_truck_dists = np.linalg.norm(
        available_customers[:, np.newaxis, :] - truck_positions[np.newaxis, :, :], axis=2
    )

    other_dists = np.copy(all_truck_dists)
    other_dists[:, active_index] = np.inf
    min_other_dists = np.min(other_dists, axis=1)

    active_advantage = all_truck_dists[:, active_index] <= ALPHA * min_other_dists

    if np.any(active_advantage):
        candidate_indices = np.where(active_advantage)[0]
        best_idx = candidate_indices[np.argmax(dist_customers_to_depot[candidate_indices])]
    else:
        best_idx = np.argmin(dist_active_to_customers)

    return int(best_idx)