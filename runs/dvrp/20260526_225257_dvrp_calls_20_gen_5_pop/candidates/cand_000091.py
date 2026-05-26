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
    
    # distances from current to customer and customer to depot
    current_to_customer = np.linalg.norm(available_customers - current_position, axis=1)
    customer_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    own_score = current_to_customer + customer_to_depot
    
    # distances to nearest other truck (excluding current)
    other_trucks = np.array([p for p in truck_positions if not np.allclose(p, current_position)])
    if len(other_trucks) == 0:
        nearest_other = np.full(len(available_customers), np.inf)
    else:
        dist_to_other = np.linalg.norm(available_customers[:, None, :] - other_trucks[None, :, :], axis=2)
        nearest_other = np.min(dist_to_other, axis=1)
    
    # waiting condition: if current truck is not farthest from depot and many customers remain
    truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_dist = np.linalg.norm(current_position - depot_position)
    max_dist = np.max(truck_depot_dists)
    if current_dist < max_dist and len(available_customers) > 2 * len(truck_positions):
        return None
    
    # Parameters with time dependency and truck-distance scaling
    beta = 1.5 / (1 + 0.1 * current_time)  # decay isolation weight
    gamma_base = 1.5 * (1 + 0.1 * current_time)  # base far-customer bias
    # scale gamma by truck's relative distance to depot
    mean_dist = np.mean(truck_depot_dists) + 1e-6
    gamma = gamma_base * (1 + current_dist / mean_dist)
    
    score = own_score - beta * nearest_other - gamma * customer_to_depot
    best_idx = np.argmin(score)
    return int(best_idx)