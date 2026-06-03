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
    
    # Compute active truck's distance to each customer and to depot after
    active_customer_dists = np.linalg.norm(available_customers - current_position, axis=1)
    customer_depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)
    active_costs = active_customer_dists + customer_depot_dists
    
    if n_trucks == 1:
        best_idx = np.argmin(active_costs)
        return int(best_idx)
    
    # Compute max distance to depot among other trucks
    other_indices = [i for i in range(n_trucks) if i != active_idx]
    other_positions = truck_positions[other_indices]
    other_depot_dists = np.linalg.norm(other_positions - depot_position, axis=1)
    other_depot_max = np.max(other_depot_dists)
    
    # For each customer, compute candidate TTT = max(active_cost, other_depot_max)
    candidate_ttt = np.maximum(active_costs, other_depot_max)
    # In case of ties, prefer smaller active_cost
    # Use lexsort: first by candidate_ttt ascending, then by active_cost ascending
    order = np.lexsort((active_costs, candidate_ttt))
    best_idx = order[0]
    return int(best_idx)