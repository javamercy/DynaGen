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
    best_score = -np.inf
    best_idx = -1
    best_this_cost = None
    alpha = 0.5  # increased penalty
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
        if score > best_score or (score == best_score and (best_this_cost is None or this_cost < best_this_cost)):
            best_score = score
            best_idx = i
            best_this_cost = this_cost
    return best_idx