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
    
    current_to_customer = np.linalg.norm(available_customers - current_position, axis=1)
    customer_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    cost_now = current_to_customer + customer_to_depot
    
    mask = np.all(np.abs(truck_positions - current_position) < 1e-8, axis=1)
    other_trucks = truck_positions[~mask]
    
    if len(other_trucks) == 0:
        best_idx = np.argmin(cost_now)
        return int(best_idx)
    
    other_to_customer = np.linalg.norm(
        available_customers[:, None, :] - other_trucks[None, :, :], axis=2
    )
    best_other_cost = np.min(other_to_customer, axis=1) + customer_to_depot
    regret = best_other_cost - cost_now
    max_regret = np.max(regret)
    
    if max_regret > 1e-6:
        best_idx = np.argmax(regret)
        return int(best_idx)
    
    current_depot_dist = np.linalg.norm(current_position - depot_position)
    other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
    min_other_depot = np.min(other_depot_dists)
    is_significantly_closer = current_depot_dist < min_other_depot * 0.8
    n_avail = len(available_customers)
    n_other = len(other_trucks)
    ratio = n_avail / n_other
    
    if is_significantly_closer and ratio <= 1.0:
        return None
    else:
        best_idx = np.argmin(cost_now)
        return int(best_idx)