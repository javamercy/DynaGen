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
    
    # Compute distances for current truck
    current_to_customer = np.linalg.norm(available_customers - current_position, axis=1)
    customer_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    cost_now = current_to_customer + customer_to_depot
    
    # Identify other trucks (exclude current position)
    mask = np.all(np.abs(truck_positions - current_position) < 1e-8, axis=1)
    other_trucks = truck_positions[~mask]
    
    if len(other_trucks) == 0:
        # Only one truck: pick min cost
        best_idx = np.argmin(cost_now)
        return int(best_idx)
    
    # Compute best other cost for each customer
    other_to_customer = np.linalg.norm(
        available_customers[:, None, :] - other_trucks[None, :, :], axis=2
    )
    best_other_cost = np.min(other_to_customer, axis=1) + customer_to_depot
    
    regret = best_other_cost - cost_now
    max_regret = np.max(regret)
    
    if max_regret > 1e-6:
        # Select customer with highest regret
        best_idx = np.argmax(regret)
        return int(best_idx)
    
    # No positive regret: consider waiting
    current_depot_dist = np.linalg.norm(current_position - depot_position)
    other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
    is_closest_to_depot = current_depot_dist < np.min(other_depot_dists) - 1e-6
    
    # Ratio of available customers to total trucks
    ratio = len(available_customers) / len(truck_positions)
    wait_threshold = 1.0  # wait if fewer customers than trucks
    
    if is_closest_to_depot and ratio < wait_threshold:
        # Wait to let other trucks serve
        return None
    else:
        # Serve the customer with minimum cost, including depot-reduction bonus
        # Increase weight on customer-to-depot distance to encourage depot approach
        beta = 0.3  # weight for depot reduction bonus
        bonus_cost = current_to_customer + (1 + beta) * customer_to_depot
        best_idx = np.argmin(bonus_cost)
        return int(best_idx)