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
    # find index of current truck (closest to current_position)
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diff)
    # precompute direct distance from current to depot
    current_to_depot = np.linalg.norm(current_position - depot_position)
    best_score = -np.inf
    best_idx = -1
    best_this_cost = np.inf
    alpha = 0.5  # penalty coefficient
    for i in range(available_customers.shape[0]):
        cust = available_customers[i]
        this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        detour = this_cost - current_to_depot  # extra distance compared to going straight to depot
        # best cost from other trucks
        other_costs = []
        for j in range(n_trucks):
            if j == current_idx:
                continue
            other_cost = np.linalg.norm(truck_positions[j] - cust) + np.linalg.norm(cust - depot_position)
            other_costs.append(other_cost)
        if len(other_costs) > 0:
            other_min = min(other_costs)
            regret = max(0, other_min - this_cost)
        else:
            regret = 0
        score = regret - alpha * detour
        if score > best_score or (score == best_score and this_cost < best_this_cost):
            best_score = score
            best_idx = i
            best_this_cost = this_cost
    return best_idx