import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    # Find index of current truck
    idx_self = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    n_trucks = truck_positions.shape[0]
    if n_trucks == 1:
        dists = np.linalg.norm(available_customers - current_position, axis=1)
        return int(np.argmin(dists))
    
    # Compute current max return time (direct to depot)
    dists_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max_return = np.max(dists_to_depot)
    
    alpha = 0.5
    best_score = -np.inf
    best_idx = -1
    
    for i, cust in enumerate(available_customers):
        d_self = np.linalg.norm(current_position - cust)
        d_self_depot = np.linalg.norm(cust - depot_position)
        self_return = d_self + d_self_depot
        
        # Distance from other trucks to this customer
        other_dists = np.linalg.norm(truck_positions - cust, axis=1)
        other_dists[idx_self] = np.inf
        min_other = np.min(other_dists)
        
        regret = min_other - d_self
        
        # Makespan increase penalty
        increase = max(0, self_return - current_max_return)
        score = regret - alpha * increase
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    if best_score > 0:
        return best_idx
    else:
        return None