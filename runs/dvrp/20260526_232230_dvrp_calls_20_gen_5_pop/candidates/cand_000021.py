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

    # compute distances from each truck to depot
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # identify active truck index
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))

    # evaluate waiting option
    returns_wait = dist_to_depot.copy()  # active's return already included
    max_ret_wait = np.max(returns_wait)
    min_ret_wait = np.min(returns_wait)
    obj_wait = max_ret_wait + 0.3 * (max_ret_wait - min_ret_wait)

    best_obj = obj_wait
    best_idx = None

    for i, cust in enumerate(available_customers):
        dist_to_cust = np.linalg.norm(current_position - cust)
        dist_cust_to_depot = np.linalg.norm(cust - depot_position)
        active_return = dist_to_cust + dist_cust_to_depot
        returns = dist_to_depot.copy()
        returns[active_idx] = active_return
        max_ret = np.max(returns)
        min_ret = np.min(returns)
        obj = max_ret + 0.3 * (max_ret - min_ret)
        if obj < best_obj:
            best_obj = obj
            best_idx = i

    return best_idx