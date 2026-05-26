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
    
    # distances
    current_to_customer = np.linalg.norm(available_customers - current_position, axis=1)
    customer_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    own_score = current_to_customer + customer_to_depot
    
    # nearest other truck penalty
    other_trucks = np.array([p for p in truck_positions if not np.allclose(p, current_position)])
    if len(other_trucks) == 0:
        nearest_other = np.full(len(available_customers), np.inf)
    else:
        dist_to_other = np.linalg.norm(available_customers[:, None, :] - other_trucks[None, :, :], axis=2)
        nearest_other = np.min(dist_to_other, axis=1)
    
    beta = 0.5
    base_score = own_score - beta * nearest_other
    
    # Dynamic depot-return pressure
    estimated_current_return = current_time + current_to_customer + customer_to_depot
    if len(other_trucks) > 0:
        estimated_others_return = current_time + np.linalg.norm(other_trucks - depot_position, axis=1)
        max_others_return = np.max(estimated_others_return)
        excess = np.maximum(0, estimated_current_return - max_others_return)
    else:
        excess = 0
    
    gamma = 1.0
    score = base_score + gamma * excess
    
    best_idx = np.argmin(score)
    return int(best_idx)