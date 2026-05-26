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
    
    # time-dependent parameters: beta (isolation) decays, gamma (depot) decays
    t = current_time
    beta = 2.0 * max(0, 1 - 0.08 * t)      # initial 2.0, decays faster
    gamma = 2.0 * max(0, 1 - 0.1 * t)       # initial 2.0 (favors far), decays to 0
    
    # score = cur_to_cust + cust_to_depot - beta*nearest_other - gamma*cust_to_depot
    # = cur_to_cust - beta*nearest_other + (1 - gamma)*cust_to_depot
    score = cur_to_cust + cust_to_depot - beta * nearest_other - gamma * cust_to_depot
    
    # tie-breaking: prefer smaller cust_to_depot if scores very close
    # we will modify argmin to consider second criterion
    best_idx = np.argmin(score)
    best_score = score[best_idx]
    # find all within epsilon of best
    eps = 1e-6
    candidates = np.where(score - best_score < eps)[0]
    if len(candidates) > 1:
        # among candidates, choose one with smallest cust_to_depot
        best_idx = candidates[np.argmin(cust_to_depot[candidates])]
    
    # waiting condition: only if very close to depot, other trucks far, and best own_score high
    cur_to_depot = np.linalg.norm(current_position - depot_position)
    mean_other_depot = np.mean(np.linalg.norm(other_trucks - depot_position, axis=1))
    min_own = np.min(own_score)
    # wait only if current truck is very near depot, others are far, and own_score is large relative to current distance
    if cur_to_depot < 0.3 * mean_other_depot and min_own > 3 * cur_to_depot:
        # also avoid waiting when many customers remain (crude proxy: len>5)
        if len(available_customers) <= 5:
            return None
    
    return int(best_idx)