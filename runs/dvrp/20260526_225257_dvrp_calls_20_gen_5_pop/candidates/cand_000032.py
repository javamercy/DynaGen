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
    
    # Cost for current truck to serve each customer: travel to customer + return to depot
    current_to_customer = np.linalg.norm(available_customers - current_position, axis=1)
    customer_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    cost_now = current_to_customer + customer_to_depot
    
    # Identify other trucks (exclude current position with a tolerance)
    mask = np.all(np.abs(truck_positions - current_position) < 1e-8, axis=1)
    other_trucks = truck_positions[~mask]
    
    if len(other_trucks) == 0:
        # Only one truck, pick customer with minimal cost
        best_idx = np.argmin(cost_now)
        return int(best_idx)
    
    # Compute best cost if served by another truck
    # other_to_customer: (n_cust, n_other)
    other_to_customer = np.linalg.norm(available_customers[:, None, :] - other_trucks[None, :, :], axis=2)
    best_other_cost = np.min(other_to_customer, axis=1) + customer_to_depot
    
    # Regret = how much better to serve now by this truck vs best other
    regret = best_other_cost - cost_now
    
    # Bonus for customers that reduce the maximum depot distance among trucks
    current_depot_dist = np.linalg.norm(current_position - depot_position)
    other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
    current_max_depot = max(current_depot_dist, np.max(other_depot_dists))
    # For each customer, potential new max if this truck serves it
    new_depot_dist = customer_to_depot  # truck would be at customer location
    other_max = np.max(other_depot_dists)
    new_max = np.maximum(new_depot_dist, other_max)
    reduction = current_max_depot - new_max
    bonus = 0.1 * np.maximum(reduction, 0)  # only positive reduction
    regret += bonus
    
    max_regret = np.max(regret)
    if max_regret > 1e-6:
        best_idx = np.argmax(regret)
        return int(best_idx)
    
    # No positive regret: all customers are as good or better served by another truck.
    # Decide whether to wait: only if this truck is not the farthest and there are more than 2 remaining customers.
    if current_depot_dist < np.max(other_depot_dists) and len(available_customers) > 2:
        return None
    else:
        # Current truck is farthest or few customers left: pick customer with minimal cost
        best_idx = np.argmin(cost_now)
        return int(best_idx)