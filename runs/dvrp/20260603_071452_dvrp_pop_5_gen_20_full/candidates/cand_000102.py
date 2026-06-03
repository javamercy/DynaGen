import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    
    n_trucks = truck_positions.shape[0]
    # Identify active truck index
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")
    
    # Distances from each customer to depot
    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)
    
    # Active truck's cost (current -> customer -> depot)
    active_dists = np.linalg.norm(available_customers - current_position, axis=1)
    active_costs = active_dists + depot_dists
    
    if n_trucks == 1:
        best_idx = np.argmin(active_costs)
        return int(best_idx)
    
    # Compute costs for other trucks (excluding active)
    mask = np.ones(n_trucks, dtype=bool)
    mask[active_idx] = False
    other_positions = truck_positions[mask]
    other_dists = np.linalg.norm(
        available_customers[:, np.newaxis, :] - other_positions[np.newaxis, :, :], axis=2
    )
    other_costs = other_dists + depot_dists[:, np.newaxis]  # (n_available, n_other)
    min_other = np.min(other_costs, axis=1)
    
    # Compute score: penalize active cost by multiplier >1
    penalty_mult = 1.2
    scores = min_other - penalty_mult * active_costs
    
    # Select customer with maximum positive score; if none, wait
    positive = scores > 0
    if np.any(positive):
        # Among positive scores, choose max score, tie-break by lowest active cost
        best_idx = np.lexsort((active_costs, -scores))[0]
        return int(best_idx)
    else:
        return None