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
    # Identify active truck index (closest position)
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    # Current distances to depot for all trucks
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # Current return times if active truck waits (i.e., returns directly)
    current_return_times = dist_to_depot.copy()
    current_max = np.max(current_return_times)
    current_min = np.min(current_return_times)
    current_bal = current_max - current_min
    current_obj = current_max + 0.1 * current_bal
    best_obj = current_obj
    best_customer = None
    for i, cust in enumerate(available_customers):
        # Active truck's return if serves this customer
        dist_to_cust = np.linalg.norm(current_position - cust)
        dist_cust_to_depot = np.linalg.norm(cust - depot_position)
        active_return = dist_to_cust + dist_cust_to_depot
        # New return times array
        new_return_times = dist_to_depot.copy()
        new_return_times[active_idx] = active_return
        new_max = np.max(new_return_times)
        new_min = np.min(new_return_times)
        new_bal = new_max - new_min
        obj = new_max + 0.1 * new_bal
        if obj < best_obj:
            best_obj = obj
            best_customer = i
    return best_customer