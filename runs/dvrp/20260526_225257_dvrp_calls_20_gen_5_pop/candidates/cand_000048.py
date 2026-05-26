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
    
    # cost_now = travel from current position to customer + customer to depot
    current_to_customer = np.linalg.norm(available_customers - current_position, axis=1)
    customer_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    cost_now = current_to_customer + customer_to_depot
    
    # Identify other trucks (exclude current truck)
    mask = np.all(np.abs(truck_positions - current_position) < 1e-8, axis=1)
    other_trucks = truck_positions[~mask]
    
    if len(other_trucks) == 0:
        best_idx = np.argmin(cost_now)
        return int(best_idx)
    
    # For each customer, compute best cost if served by another truck
    other_to_customer = np.linalg.norm(available_customers[:, None, :] - other_trucks[None, :, :], axis=2)  # (n_cust, n_other)
    best_other_cost = np.min(other_to_customer, axis=1) + customer_to_depot
    
    regret = best_other_cost - cost_now  # positive means this truck should serve it now
    
    max_regret = np.max(regret)
    if max_regret > 1e-6:
        best_idx = np.argmax(regret)
        return int(best_idx)
    
    # No positive regret -> decide whether to wait
    current_depot_dist = np.linalg.norm(current_position - depot_position)
    other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
    max_other_dist = np.max(other_depot_dists)
    
    # Threshold: wait only if current truck is not farthest AND number of customers > 2
    if current_depot_dist < max_other_dist and len(available_customers) > 2:
        return None
    else:
        best_idx = np.argmin(cost_now)
        return int(best_idx)