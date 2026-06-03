import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    # Find current truck index
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diff)
    # Precompute other trucks' distance to depot
    other_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    best_max = np.inf
    best_idx = -1
    best_this_cost = np.inf
    for i, cust in enumerate(available_customers):
        this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        this_max = this_cost
        for j, dist in enumerate(other_to_depot):
            if j == current_idx:
                continue
            if dist > this_max:
                this_max = dist
        if this_max < best_max or (this_max == best_max and this_cost < best_this_cost):
            best_max = this_max
            best_idx = i
            best_this_cost = this_cost
    return best_idx