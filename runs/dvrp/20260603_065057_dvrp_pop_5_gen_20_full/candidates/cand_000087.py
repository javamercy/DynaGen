import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    n_trucks = truck_positions.shape[0]
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diff)
    current_dist_to_depot = dist_to_depot[current_idx]
    max_dist_to_depot = np.max(dist_to_depot)
    if max_dist_to_depot > 0:
        penalty_weight = 0.5 * (current_dist_to_depot / max_dist_to_depot)
    else:
        penalty_weight = 0.0

    best_score = -np.inf
    best_idx = -1
    best_this_cost = np.inf
    for i, cust in enumerate(available_customers):
        this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        min_other = np.inf
        for j in range(n_trucks):
            if j == current_idx:
                continue
            other_cost = np.linalg.norm(truck_positions[j] - cust) + np.linalg.norm(cust - depot_position)
            if other_cost < min_other:
                min_other = other_cost
        if min_other == np.inf:
            min_other = 0.0
        regret = max(0, min_other - this_cost)
        score = regret - penalty_weight * this_cost
        if score > best_score or (score == best_score and this_cost < best_this_cost):
            best_score = score
            best_idx = i
            best_this_cost = this_cost
    return best_idx