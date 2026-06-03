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
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diff)
    # other trucks' direct-to-depot distances
    other_depot = np.linalg.norm(np.delete(truck_positions, current_idx, axis=0) - depot_position, axis=1)
    other_max = np.max(other_depot) if len(other_depot) > 0 else 0.0
    best_max = np.inf
    best_idx = -1
    best_this = np.inf
    for i, cust in enumerate(available_customers):
        this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        new_max = max(this_cost, other_max)
        if new_max < best_max or (new_max == best_max and this_cost < best_this):
            best_max = new_max
            best_idx = i
            best_this = this_cost
    return best_idx