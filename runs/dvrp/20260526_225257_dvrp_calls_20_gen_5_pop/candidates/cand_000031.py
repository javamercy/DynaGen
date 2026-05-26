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
    
    # cost for this truck: travel to customer + return to depot
    current_to_customer = np.linalg.norm(available_customers - current_position, axis=1)
    customer_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    cost_now = current_to_customer + customer_to_depot
    
    # identify other trucks (exclude current position)
    mask = np.all(np.abs(truck_positions - current_position) < 1e-8, axis=1)
    other_trucks = truck_positions[~mask]
    
    if len(other_trucks) == 0:
        best_idx = np.argmin(cost_now)
        return int(best_idx)
    
    # best other cost for each customer
    other_to_customer = np.linalg.norm(
        available_customers[:, None, :] - other_trucks[None, :, :], axis=2
    )  # (n_cust, n_other)
    best_other_cost = np.min(other_to_customer, axis=1) + customer_to_depot
    
    regret = best_other_cost - cost_now  # positive means urgent
    max_regret = np.max(regret)
    
    if max_regret > 1e-6:
        best_idx = np.argmax(regret)
        return int(best_idx)
    
    # No positive regret
    current_depot_dist = np.linalg.norm(current_position - depot_position)
    other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
    max_other_depot = np.max(other_depot_dists) if len(other_depot_dists) > 0 else 0.0
    is_farthest = current_depot_dist >= max_other_depot - 1e-6
    
    if is_farthest:
        # farthest truck should not wait
        best_idx = np.argmin(cost_now)
        return int(best_idx)
    else:
        min_cost = np.min(cost_now)
        if min_cost < current_depot_dist:
            # nearby customer worth serving
            best_idx = np.argmin(cost_now)
            return int(best_idx)
        elif len(available_customers) > 2:
            # wait (other trucks are better positioned)
            return None
        else:
            # few customers left, serve to avoid idle
            best_idx = np.argmin(cost_now)
            return int(best_idx)