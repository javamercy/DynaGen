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
    # Single truck case: always serve, minimize total travel to customer and back to depot
    if n_trucks == 1:
        d_curr = np.linalg.norm(available_customers - current_position, axis=1)
        cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
        scores = d_curr + cust_to_depot
        return int(np.argmin(scores))
    # Multiple trucks
    # Compute distances from each truck to depot
    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # Identify current truck index (by closest match)
    current_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    current_to_depot = truck_to_depot[current_idx]
    # Distances from other trucks to depot
    other_depot_dists = np.delete(truck_to_depot, current_idx)
    current_max = max(current_to_depot, np.max(other_depot_dists))
    
    # For each customer, compute new estimated return for current truck and new max
    d_curr = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    new_est_return = d_curr + cust_to_depot
    new_max = np.maximum(np.max(other_depot_dists), new_est_return)
    best_idx = int(np.argmin(new_max))
    min_new_max = new_max[best_idx]
    # If no customer reduces the current max, wait
    if min_new_max < current_max:
        return best_idx
    else:
        return None