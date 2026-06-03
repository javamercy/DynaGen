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
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diff)
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    avg_dist = np.mean(dist_to_depot)
    curr_dist = dist_to_depot[current_idx]
    burden = curr_dist / (avg_dist + 1e-6)
    lam = 0.1
    best_score = -np.inf
    best_idx = -1
    best_this_cost = np.inf
    for i in range(available_customers.shape[0]):
        cust = available_customers[i]
        this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        other_costs = []
        for j in range(n_trucks):
            if j == current_idx:
                continue
            other_cost = np.linalg.norm(truck_positions[j] - cust) + np.linalg.norm(cust - depot_position)
            other_costs.append(other_cost)
        if other_costs:
            other_min = min(other_costs)
            regret = max(0, other_min - this_cost)
        else:
            regret = 0.0
        penalty = lam * max(0, burden - 1) * this_cost
        score = regret - penalty
        if score > best_score or (score == best_score and this_cost < best_this_cost):
            best_score = score
            best_idx = i
            best_this_cost = this_cost
    return best_idx