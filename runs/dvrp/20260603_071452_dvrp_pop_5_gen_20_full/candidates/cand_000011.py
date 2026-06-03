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
    # find active truck index
    active_idx = None
    for i, pos in enumerate(truck_positions):
        if np.allclose(pos, current_position):
            active_idx = i
            break
    if active_idx is None:
        # fallback: assume first
        active_idx = 0
    
    # single truck case: always serve nearest
    if n_trucks == 1:
        dists = np.linalg.norm(available_customers - current_position, axis=1)
        best_idx = int(np.argmin(dists))
        return best_idx
    
    # compute other trucks' return distances
    other_returns = []
    for j in range(n_trucks):
        if j != active_idx:
            other_returns.append(np.linalg.norm(truck_positions[j] - depot_position))
    max_other = max(other_returns) if other_returns else 0.0
    
    wait_makespan = max(np.linalg.norm(current_position - depot_position), max_other)
    
    best_customer = None
    best_makespan = float('inf')
    best_active_dist = float('inf')
    
    for i, cust in enumerate(available_customers):
        active_dist = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        makespan = max(active_dist, max_other)
        if makespan < best_makespan or (makespan == best_makespan and active_dist < best_active_dist):
            best_makespan = makespan
            best_active_dist = active_dist
            best_customer = i
    
    if best_customer is None:
        return None
    if best_makespan >= wait_makespan:
        return None
    else:
        return best_customer