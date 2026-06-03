import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    n_trucks = truck_positions.shape[0]
    dist_to_current = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(dist_to_current)
    best_score = -np.inf
    best_idx = -1
    best_dist_to_cust = None
    alpha = 0.1
    beta = 2.0
    for i, cust in enumerate(available_customers):
        dist_to_cust = np.linalg.norm(current_position - cust)
        dist_to_depot = np.linalg.norm(cust - depot_position)
        this_cost = dist_to_cust + dist_to_depot
        other_costs = [
            np.linalg.norm(truck_positions[j] - cust) + dist_to_depot
            for j in range(n_trucks) if j != current_idx
        ]
        if other_costs:
            other_min = min(other_costs)
            regret = max(0, other_min - this_cost)
        else:
            regret = 0
        score = regret - alpha * (dist_to_cust + beta * dist_to_depot)
        if score > best_score or (score == best_score and (best_dist_to_cust is None or dist_to_cust < best_dist_to_cust)):
            best_score = score
            best_idx = i
            best_dist_to_cust = dist_to_cust
    return best_idx