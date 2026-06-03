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
    # Identify the current truck index
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diff)
    # Compute alpha based on imbalance of truck-to-depot distances
    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    max_dist = np.max(truck_to_depot)
    mean_dist = np.mean(truck_to_depot)
    if mean_dist > 1e-6:
        alpha = 0.1 * (1 + (max_dist - mean_dist) / mean_dist)
    else:
        alpha = 0.1
    best_score = -np.inf
    best_idx = -1
    for i, cust in enumerate(available_customers):
        this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        other_costs = [
            np.linalg.norm(truck_positions[j] - cust) + np.linalg.norm(cust - depot_position)
            for j in range(n_trucks) if j != current_idx
        ]
        if other_costs:
            other_min = min(other_costs)
            regret = max(0, other_min - this_cost)
        else:
            regret = 0
        score = regret - alpha * this_cost
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx if best_idx != -1 else None