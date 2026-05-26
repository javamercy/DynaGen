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
    
    n_trucks = len(truck_positions)
    n_remaining = len(available_customers)
    
    # Weight for depot return distance: increases with remaining work per truck
    weight = 1 + 0.5 * (n_remaining / n_trucks)
    
    # Compute cost_now for this truck with weighted return
    current_to_customer = np.linalg.norm(available_customers - current_position, axis=1)
    customer_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    cost_now = current_to_customer + weight * customer_to_depot
    
    # Identify other trucks
    mask = np.all(np.abs(truck_positions - current_position) < 1e-8, axis=1)
    other_trucks = truck_positions[~mask]
    
    if len(other_trucks) == 0:
        best_idx = np.argmin(cost_now)
        return int(best_idx)
    
    # Best other cost for each customer with same weight
    other_to_customer = np.linalg.norm(
        available_customers[:, None, :] - other_trucks[None, :, :], axis=2
    )
    best_other_cost = np.min(other_to_customer, axis=1) + weight * customer_to_depot
    
    regret = best_other_cost - cost_now
    max_regret = np.max(regret)
    
    if max_regret > 1e-6:
        best_idx = np.argmax(regret)
        return int(best_idx)
    
    # No positive regret: decide whether to wait
    current_depot_dist = np.linalg.norm(current_position - depot_position)
    other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
    is_closest_to_depot = current_depot_dist < np.min(other_depot_dists) - 1e-6
    
    # Wait if closest to depot and remaining customers per truck ratio <= 0.5
    ratio = n_remaining / n_trucks
    if is_closest_to_depot and ratio <= 0.5:
        return None
    else:
        best_idx = np.argmin(cost_now)
        return int(best_idx)