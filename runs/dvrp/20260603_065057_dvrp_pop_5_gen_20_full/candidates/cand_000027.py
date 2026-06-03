import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    # Identify current truck index
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diff)
    depot = depot_position
    # Current distances from other trucks to depot
    other_dist_to_depot = [np.linalg.norm(truck_positions[j] - depot) for j in range(truck_positions.shape[0]) if j != current_idx]
    max_other = max(other_dist_to_depot) if other_dist_to_depot else -np.inf
    # Current truck distance to depot
    current_dist_to_depot = np.linalg.norm(current_position - depot)
    current_max = max(current_dist_to_depot, max_other)
    best_new_max = np.inf
    best_idx = -1
    best_this_cost = np.inf
    for i, cust in enumerate(available_customers):
        new_current_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot)
        new_max = max(new_current_return, max_other)
        if new_max < best_new_max or (new_max == best_new_max and new_current_return < best_this_cost):
            best_new_max = new_max
            best_this_cost = new_current_return
            best_idx = i
    return best_idx