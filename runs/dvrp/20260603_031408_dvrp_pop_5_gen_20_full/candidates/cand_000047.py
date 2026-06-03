import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None

    n_avail = available_customers.shape[0]
    # Distances from each truck to depot
    all_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # Find index of active truck
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    # Max other truck distance to depot
    other_mask = np.ones(truck_positions.shape[0], dtype=bool)
    other_mask[active_idx] = False
    max_other = np.max(all_to_depot[other_mask])
    # Active truck distance to depot
    dist_active_to_depot = np.linalg.norm(current_position - depot_position)

    # For each available customer
    dist_active_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    new_return_times = dist_active_to_cust + cust_to_depot
    candidate_max = np.maximum(new_return_times, max_other)

    # Nearest neighbor distances among available customers
    nn_dist = np.full(n_avail, np.inf)
    for i in range(n_avail):
        for j in range(n_avail):
            if i != j:
                d = np.linalg.norm(available_customers[i] - available_customers[j])
                if d < nn_dist[i]:
                    nn_dist[i] = d
    # If only one customer, nn_dist remains inf; set to 0 for safety
    if n_avail == 1:
        nn_dist[0] = 0.0

    # Scores: minimize candidate_max - 0.5 * nn_dist + 0.5 * cust_to_depot
    w_iso = 0.5
    w_depot = 0.5
    scores = candidate_max - w_iso * nn_dist + w_depot * cust_to_depot

    # Waiting condition
    min_candidate_max = np.min(candidate_max)
    threshold_close = 0.15 * max_other
    threshold_far = 1.15 * max_other
    if max_other > 1e-9 and dist_active_to_depot < threshold_close and min_candidate_max > threshold_far:
        return None

    # Choose customer with minimal score
    best_idx = np.argmin(scores)
    return int(best_idx)