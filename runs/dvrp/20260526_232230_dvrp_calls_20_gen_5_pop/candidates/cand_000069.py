import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if len(available_customers) == 0:
        return None
    # Distances from each truck to depot (current estimated return time if idle)
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # Identify active truck (closest to current_position)
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    current_max_return = np.max(dist_to_depot)
    best_customer = None
    best_score = np.inf
    best_active_return = np.inf
    alpha = 1.0  # penalty weight for increase
    for i, cust in enumerate(available_customers):
        # Active truck's return time if it goes to this customer
        active_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # Build new return times array
        new_return_times = dist_to_depot.copy()
        new_return_times[active_idx] = active_return
        new_max = np.max(new_return_times)
        increase = new_max - current_max_return
        # Penalize increase, even if negative (which reduces max) we don't reward extra
        score = new_max + alpha * max(0, increase)
        # Tie-break by own return time
        if score < best_score or (score == best_score and active_return < best_active_return):
            best_score = score
            best_active_return = active_return
            best_customer = i
    return best_customer