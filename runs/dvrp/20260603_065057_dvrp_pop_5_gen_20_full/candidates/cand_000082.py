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
    other_depot_dists = np.array([np.linalg.norm(truck_positions[j] - depot_position) for j in range(n_trucks) if j != current_idx])
    max_other_depot = np.max(other_depot_dists) if other_depot_dists.size > 0 else 0
    alpha = 0.5
    beta = 0.5
    best_score = -np.inf
    best_idx = -1
    best_this_cost = None
    for i, cust in enumerate(available_customers):
        this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        other_costs = [np.linalg.norm(truck_positions[j] - cust) + np.linalg.norm(cust - depot_position) for j in range(n_trucks) if j != current_idx]
        if other_costs:
            other_min = min(other_costs)
            regret = max(0, other_min - this_cost)
        else:
            regret = 0
        bottleneck_increase = max(0, this_cost - max_other_depot)
        score = regret - alpha * this_cost - beta * bottleneck_increase
        if score > best_score or (score == best_score and (best_this_cost is None or this_cost < best_this_cost)):
            best_score = score
            best_idx = i
            best_this_cost = this_cost
    return best_idx