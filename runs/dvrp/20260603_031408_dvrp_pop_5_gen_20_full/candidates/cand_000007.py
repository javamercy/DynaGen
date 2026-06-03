import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None

    # Distance from each truck to depot
    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    max_other_truck = np.max(truck_to_depot)

    # For each available customer, compute new return time for active truck
    dist_active_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    dist_cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    new_return_times = dist_active_to_cust + dist_cust_to_depot

    candidate_max = np.maximum(new_return_times, max_other_truck)
    min_candidate_max = np.min(candidate_max)
    ties = np.where(candidate_max == min_candidate_max)[0]

    if len(ties) == 1:
        return int(ties[0])

    # Tie-breaking: among ties, pick isolated customer (largest nearest-neighbor distance)
    best_idx = ties[0]
    best_nn_dist = -np.inf
    n_avail = len(available_customers)
    for idx in ties:
        # compute nearest neighbor distance excluding itself
        nn_dist = np.inf
        for j in range(n_avail):
            if j != idx:
                dd = np.linalg.norm(available_customers[idx] - available_customers[j])
                if dd < nn_dist:
                    nn_dist = dd
        # if only one customer, nn_dist stays inf, but then ties length 1, so we won't get here
        if nn_dist > best_nn_dist:
            best_nn_dist = nn_dist
            best_idx = idx
    return int(best_idx)