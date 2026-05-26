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
    
    # Find index of the deciding truck
    mask = np.all(truck_positions == current_position, axis=1)
    if np.any(mask):
        decided_idx = np.where(mask)[0][0]
    else:
        decided_idx = 0  # fallback
    
    # Distances from other trucks to depot
    other_positions = []
    other_dists = []
    for i, pos in enumerate(truck_positions):
        if i != decided_idx:
            dist = np.linalg.norm(pos - depot_position)
            other_dists.append(dist)
            other_positions.append(pos)
    other_positions = np.array(other_positions) if other_positions else np.empty((0, 2))
    other_max = max(other_dists) if other_dists else 0.0
    
    best_idx = None
    best_score = float('inf')
    best_own = float('inf')
    beta = 0.5
    
    for i, cust in enumerate(available_customers):
        own_route = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        if len(other_positions) > 0:
            nearest_other = np.min(np.linalg.norm(other_positions - cust, axis=1))
        else:
            nearest_other = 0.0
        score = max(own_route, other_max) - beta * nearest_other
        if score < best_score or (score == best_score and own_route < best_own):
            best_score = score
            best_own = own_route
            best_idx = i
    
    return best_idx