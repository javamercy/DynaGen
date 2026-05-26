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
    
    current_to_customer = np.linalg.norm(available_customers - current_position, axis=1)
    customer_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    own_score = current_to_customer + customer_to_depot
    
    other_trucks = np.array([p for idx, p in enumerate(truck_positions) if not np.allclose(p, current_position)])
    if len(other_trucks) == 0:
        # no other trucks, just pick the best own_score
        best_idx = np.argmin(own_score)
        return int(best_idx)
    
    dist_to_other = np.linalg.norm(available_customers[:, None, :] - other_trucks[None, :, :], axis=2)
    nearest_other = np.min(dist_to_other, axis=1)
    
    t = current_time
    beta = 1.5 * max(0, 1 - 0.1 * t)
    gamma = 0.3 * (1 + 0.2 * t)
    
    score = own_score - beta * nearest_other - gamma * customer_to_depot
    
    # waiting condition (original)
    current_to_depot = np.linalg.norm(current_position - depot_position)
    other_depot_dist = np.linalg.norm(other_trucks - depot_position, axis=1)
    mean_other_depot = np.mean(other_depot_dist)
    min_own = np.min(own_score)
    wait_condition = (current_to_depot < 0.5 * mean_other_depot) and (min_own > 2 * current_to_depot)
    if wait_condition and np.random.rand() < 0.5:
        return None
    
    # epsilon-greedy and softmax
    epsilon = max(0.05, 0.2 * max(0, 1 - 0.1 * t))
    if np.random.rand() < epsilon:
        return int(np.random.randint(len(available_customers)))
    
    temperature = max(0.1, 1 - 0.05 * t)
    # stabilize softmax
    scores_min = np.min(score)
    exp_scores = np.exp(-(score - scores_min) / temperature)
    probs = exp_scores / np.sum(exp_scores)
    idx = np.random.choice(len(available_customers), p=probs)
    return int(idx)