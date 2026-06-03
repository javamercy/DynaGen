import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    # identify current truck index
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diff)
    # distances of other trucks to depot
    depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    other_max_depot = np.max(np.delete(depot_dists, current_idx))
    best_max = np.inf
    best_cost = np.inf
    best_idx = -1
    for i in range(available_customers.shape[0]):
        cust = available_customers[i]
        this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        candidate_max = max(this_cost, other_max_depot)
        if (candidate_max < best_max) or (candidate_max == best_max and this_cost < best_cost):
            best_max = candidate_max
            best_cost = this_cost
            best_idx = i
    return best_idx