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
    # identify current truck index
    dist_to_current = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(dist_to_current)
    # distances from each truck to depot
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # if only one truck, just minimize new distance
    if n_trucks == 1:
        new_current_dists = np.linalg.norm(current_position - available_customers, axis=1) + np.linalg.norm(available_customers - depot_position, axis=1)
        best_idx = np.argmin(new_current_dists)
        return int(best_idx)
    # maximum distance among other trucks to depot (constant)
    max_other = np.max(np.delete(dist_to_depot, current_idx))
    new_current_dists = np.linalg.norm(current_position - available_customers, axis=1) + np.linalg.norm(available_customers - depot_position, axis=1)
    new_maxes = np.maximum(new_current_dists, max_other)
    best_max = np.min(new_maxes)
    candidates = np.where(new_maxes == best_max)[0]
    # tie-break: choose customer with smallest distance from customer to depot
    if len(candidates) > 1:
        cust_to_depot = np.linalg.norm(available_customers[candidates] - depot_position, axis=1)
        best_idx = candidates[np.argmin(cust_to_depot)]
    else:
        best_idx = candidates[0]
    return int(best_idx)