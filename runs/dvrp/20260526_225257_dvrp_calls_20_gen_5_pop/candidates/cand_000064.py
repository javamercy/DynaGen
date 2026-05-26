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
    
    # Cost for this truck: travel to customer + return to depot
    current_to_customer = np.linalg.norm(available_customers - current_position, axis=1)
    customer_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    cost_now = current_to_customer + customer_to_depot
    
    # Identify other trucks (exclude current position)
    mask = np.all(np.abs(truck_positions - current_position) < 1e-8, axis=1)
    other_trucks = truck_positions[~mask]
    
    if len(other_trucks) == 0:
        best_idx = np.argmin(cost_now)
        return int(best_idx)
    
    # Best other cost for each customer
    other_to_customer = np.linalg.norm(
        available_customers[:, None, :] - other_trucks[None, :, :], axis=2
    )
    best_other_cost = np.min(other_to_customer, axis=1) + customer_to_depot
    
    regret = best_other_cost - cost_now
    max_regret = np.max(regret)
    
    if max_regret > 1e-6:
        best_idx = np.argmax(regret)
        return int(best_idx)
    
    # No positive regret: consider waiting
    current_depot_dist = np.linalg.norm(current_position - depot_position)
    other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
    is_closest_to_depot = current_depot_dist < np.min(other_depot_dists) - 1e-6
    
    n_other = len(other_trucks)
    n_available = len(available_customers)
    wait_threshold = int(1.5 * n_other)  # Allow waiting when customers <= 1.5 * other trucks
    
    if is_closest_to_depot and n_available <= wait_threshold:
        return None
    else:
        # Serve the customer with minimum cost
        best_idx = np.argmin(cost_now)
        return int(best_idx)