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
    cur_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    own_score = cur_to_cust + cust_to_depot
    
    # other trucks
    mask = ~np.all(truck_positions == current_position, axis=1)
    other_trucks = truck_positions[mask]
    if len(other_trucks) == 0:
        return int(np.argmin(own_score))
    
    # distance from each customer to nearest other truck
    dist_to_other = np.linalg.norm(available_customers[:, None] - other_trucks[None, :], axis=2)
    nearest_other = np.min(dist_to_other, axis=1)
    
    # fleet balance: distance from customer to centroid of other trucks
    centroid_other = np.mean(other_trucks, axis=0)
    dist_to_centroid = np.linalg.norm(available_customers - centroid_other, axis=1)
    
    # time-dependent parameters
    t = current_time
    beta = 2.0 * max(0, 1 - 0.12 * t)  # faster decay
    gamma = 2.0 * max(0, 1 - 0.1 * t)
    alpha = 0.4  # fleet balance weight
    
    # score = cur_to_cust - beta*nearest_other + (1-gamma)*cust_to_depot - alpha*dist_to_centroid
    score = cur_to_cust + cust_to_depot - beta * nearest_other - gamma * cust_to_depot - alpha * dist_to_centroid
    
    # tie-breaking: prefer smaller cust_to_depot if scores very close
    best_idx = np.argmin(score)
    best_score = score[best_idx]
    eps = 1e-6
    candidates = np.where(score - best_score < eps)[0]
    if len(candidates) > 1:
        best_idx = candidates[np.argmin(cust_to_depot[candidates])]
    
    # waiting condition: stricter
    cur_to_depot = np.linalg.norm(current_position - depot_position)
    mean_other_depot = np.mean(np.linalg.norm(other_trucks - depot_position, axis=1))
    min_own = np.min(own_score)
    if cur_to_depot < 0.2 * mean_other_depot and min_own > 4 * cur_to_depot and len(available_customers) <= 3:
        return None
    
    return int(best_idx)