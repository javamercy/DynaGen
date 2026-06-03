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
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")
    
    # Distance from each truck to depot
    truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    
    def dist(p1, p2):
        return np.linalg.norm(p1 - p2)
    
    if n_trucks == 1:
        # Single truck: always assign customer with smallest active_time
        best_idx = None
        best_active = np.inf
        for i, cust in enumerate(available_customers):
            active_time = dist(current_position, cust) + dist(cust, depot_position)
            if active_time < best_active:
                best_active = active_time
                best_idx = i
        return best_idx
    
    # Current maximum distance to depot (proxy for current max completion time)
    current_max = np.max(truck_depot_dists)
    
    best_idx = None
    best_max = np.inf
    best_active = np.inf
    
    for i, cust in enumerate(available_customers):
        active_time = dist(current_position, cust) + dist(cust, depot_position)
        # other trucks' distances to depot (assuming they stay put)
        other_dists = [truck_depot_dists[j] for j in range(n_trucks) if j != active_idx]
        potential_max = max(active_time, max(other_dists)) if other_dists else active_time
        
        if potential_max < best_max or (potential_max == best_max and active_time < best_active):
            best_max = potential_max
            best_active = active_time
            best_idx = i
    
    # Wait if the best assignment increases the current max
    if best_max > current_max:
        return None
    else:
        return best_idx