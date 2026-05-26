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
    
    # Costs for current truck
    current_to_customer = np.linalg.norm(available_customers - current_position, axis=1)
    customer_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    cost_now = current_to_customer + customer_to_depot
    
    # Identify other trucks
    mask = np.all(np.abs(truck_positions - current_position) < 1e-8, axis=1)
    other_trucks = truck_positions[~mask]
    
    if len(other_trucks) == 0:
        best_idx = np.argmin(cost_now)
        return int(best_idx)
    
    # Other trucks' costs to each customer
    other_to_customer = np.linalg.norm(available_customers[:, None, :] - other_trucks[None, :, :], axis=2)
    best_other_cost = np.min(other_to_customer, axis=1) + customer_to_depot
    
    # Regret
    regret = best_other_cost - cost_now
    
    # Bonus for reducing max depot distance
    current_depot_dist = np.linalg.norm(current_position - depot_position)
    other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
    current_max_depot = max(current_depot_dist, np.max(other_depot_dists))
    # New max if this truck serves customer: max(other_depot_dists.max(), customer_to_depot)
    new_max = np.maximum(np.max(other_depot_dists), customer_to_depot)
    reduction = current_max_depot - new_max
    # Dynamic scaling: bonus increases when few customers remain
    n_avail = len(available_customers)
    scaling = 1.0 + 0.5 / (n_avail + 1.0)  # ranges from 1.0 to ~1.5
    bonus = 0.5 * reduction * scaling
    regret += bonus
    
    max_regret = np.max(regret)
    if max_regret > 1e-6:
        best_idx = np.argmax(regret)
        return int(best_idx)
    
    # Wait condition: wait only if no positive regret and current truck is not farthest
    if current_depot_dist < np.max(other_depot_dists):
        return None
    else:
        best_idx = np.argmin(cost_now)
        return int(best_idx)