import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    
    cur = current_position
    avail = available_customers
    n = len(avail)
    
    dist_to_cust = np.linalg.norm(avail - cur, axis=1)
    cust_to_depot = np.linalg.norm(avail - depot_position, axis=1)
    active_return = dist_to_cust + cust_to_depot
    
    # Compute nearest other distance for each customer
    if n == 1:
        min_nn = np.zeros(1)
    else:
        from scipy.spatial.distance import cdist
        all_dists = cdist(avail, avail)
        np.fill_diagonal(all_dists, np.inf)
        min_nn = np.min(all_dists, axis=1)
    
    # Compute overall max distance from trucks to depot (including active truck's current distance)
    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    overall_max = np.max(truck_to_depot)
    candidate_max = np.maximum(active_return, overall_max)
    
    # Composite score: combine parent1 (min_nn - dist_to_cust) and parent3 (-candidate_max)
    # Weighted: alpha * (min_nn - dist_to_cust) - beta * candidate_max
    alpha = 1.0
    beta = 1.0
    scores = alpha * (min_nn - dist_to_cust) - beta * candidate_max
    
    # Softmax probabilities with temperature
    temp = 0.5
    exp_scores = np.exp(scores / temp)
    probs = exp_scores / np.sum(exp_scores)
    
    # Sample
    idx = np.random.choice(n, p=probs)
    return int(idx)