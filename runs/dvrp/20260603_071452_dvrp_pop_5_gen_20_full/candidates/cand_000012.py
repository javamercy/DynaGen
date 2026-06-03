import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    
    n_trucks = len(truck_positions)
    # Find active truck index
    active_idx = None
    for i in range(n_trucks):
        if np.array_equal(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")
    
    # Compute distances from each truck to depot
    truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max = np.max(truck_depot_dists)
    
    # Precompute depot distances for customers
    cust_depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)
    
    # Active truck costs
    active_costs = np.linalg.norm(current_position - available_customers, axis=1) + cust_depot_dists
    
    # Single truck case: always serve the customer with minimum active cost
    if n_trucks == 1:
        return int(np.argmin(active_costs))
    
    # Compute minimum cost among other trucks for each customer
    min_other = np.full(len(available_customers), np.inf)
    for j in range(n_trucks):
        if j == active_idx:
            continue
        other_cost = np.linalg.norm(truck_positions[j] - available_customers, axis=1) + cust_depot_dists
        min_other = np.minimum(min_other, other_cost)
    
    # Regret condition: active cost <= best other cost (with small tolerance)
    mask = active_costs <= min_other + 1e-9
    if not np.any(mask):
        return None
    
    # Maximum return time among other trucks (current, since they haven't moved)
    other_max = np.max(truck_depot_dists[np.arange(n_trucks) != active_idx])
    # New max if active truck serves customer
    new_maxes = np.maximum(active_costs, other_max)
    
    # Only consider candidates
    candidate_indices = np.where(mask)[0]
    candidate_new_max = new_maxes[mask]
    candidate_active = active_costs[mask]
    
    # Find minimum new max among candidates
    best_new_max = np.min(candidate_new_max)
    # Tie-break: among those with best new max, smallest active cost
    tie_mask = candidate_new_max == best_new_max
    tie_indices = candidate_indices[tie_mask]
    best_idx = tie_indices[np.argmin(candidate_active[tie_mask])]
    
    # Wait if serving would increase current maximum
    if best_new_max > current_max + 1e-9:
        return None
    
    return int(best_idx)