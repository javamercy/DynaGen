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
    # Identify current truck index
    dist_to_current = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(dist_to_current)
    # Compute other trucks' distances to depot
    other_dist_to_depot = [np.linalg.norm(truck_positions[j] - depot_position) for j in range(n_trucks) if j != current_idx]
    best_idx = None
    best_max = float('inf')
    best_this_cost = float('inf')
    for i in range(available_customers.shape[0]):
        cust = available_customers[i]
        this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        if other_dist_to_depot:
            max_ret = max(this_cost, max(other_dist_to_depot))
        else:
            max_ret = this_cost
        if max_ret < best_max or (max_ret == best_max and this_cost < best_this_cost):
            best_max = max_ret
            best_this_cost = this_cost
            best_idx = i
    return best_idx