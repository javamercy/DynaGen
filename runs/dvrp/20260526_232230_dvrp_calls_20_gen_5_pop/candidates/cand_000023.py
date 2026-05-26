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
    # Find active truck index
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    # Distances from all trucks to depot
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # Waiting option: no change to active truck
    best_max = np.max(dist_to_depot)
    best_sum = np.sum(dist_to_depot)
    best_customer = None
    # Evaluate each customer
    for i, cust in enumerate(available_customers):
        dist_to_cust = np.linalg.norm(current_position - cust)
        dist_cust_to_depot = np.linalg.norm(cust - depot_position)
        active_return = dist_to_cust + dist_cust_to_depot
        return_times = dist_to_depot.copy()
        return_times[active_idx] = active_return
        max_ret = np.max(return_times)
        sum_ret = np.sum(return_times)
        # Compare: primary objective is max, secondary is sum
        if max_ret < best_max or (max_ret == best_max and sum_ret < best_sum):
            best_max = max_ret
            best_sum = sum_ret
            best_customer = i
    return best_customer