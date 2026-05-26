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
    # Distances from each truck to depot
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # Active truck index (closest to current_position)
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    best_customer = None
    best_score = np.inf
    alpha = 0.3  # weight for gap between max and second-max
    beta = 0.1   # weight for mean return
    for i, cust in enumerate(available_customers):
        # Active truck's estimated return if it goes to this customer
        active_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # Build return times array with active truck's value replaced
        return_times = dist_to_depot.copy()
        return_times[active_idx] = active_return
        # Sort descending to easily get max and second max
        sorted_times = np.sort(return_times)[::-1]
        max_return = sorted_times[0]
        second_max = sorted_times[1] if len(sorted_times) > 1 else max_return
        gap = max_return - second_max
        mean_return = np.mean(return_times)
        score = max_return + alpha * gap + beta * mean_return
        if score < best_score:
            best_score = score
            best_customer = i
    return best_customer