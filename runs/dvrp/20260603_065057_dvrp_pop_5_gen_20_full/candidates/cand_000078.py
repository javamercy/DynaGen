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
    other_mask = np.ones(n_trucks, dtype=bool)
    other_mask[current_idx] = False
    other_truck_positions = truck_positions[other_mask]
    if len(other_truck_positions) > 0:
        M = np.max(np.linalg.norm(other_truck_positions - depot_position, axis=1))
    else:
        M = -np.inf
    best_score = -np.inf
    best_customer_idx = -1
    best_this_cost = np.inf
    for i in range(available_customers.shape[0]):
        cust = available_customers[i]
        this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        other_costs = np.array([
            np.linalg.norm(truck_positions[j] - cust) + np.linalg.norm(cust - depot_position)
            for j in range(n_trucks) if j != current_idx
        ])
        if len(other_costs) > 0:
            other_min = np.min(other_costs)
            regret = max(0, other_min - this_cost)
        else:
            regret = 0.0
        penalty = max(0, this_cost - M)
        alpha = 1.0
        score = regret - alpha * penalty
        if score > best_score or (score == best_score and this_cost < best_this_cost):
            best_score = score
            best_customer_idx = i
            best_this_cost = this_cost
    return best_customer_idx