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
    # Find index of current truck (closest to current_position)
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diff)
    # Other trucks' estimated remaining distance (straight to depot)
    other_est = np.linalg.norm(truck_positions - depot_position, axis=1)
    other_est[current_idx] = -1  # ignore current truck
    max_other = np.max(other_est)
    best_score = -np.inf
    best_idx = -1
    best_this_cost = np.inf
    for i, cust in enumerate(available_customers):
        this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        potential_max = max(this_cost, max_other)
        score = -potential_max  # higher is better (minimize max)
        if score > best_score or (score == best_score and this_cost < best_this_cost):
            best_score = score
            best_idx = i
            best_this_cost = this_cost
    return best_idx