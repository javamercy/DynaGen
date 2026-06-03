import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    # Find index of current truck in truck_positions
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diff)
    best_regret = -np.inf
    best_customer_idx = -1
    best_this_cost = np.inf
    n_trucks = truck_positions.shape[0]
    for i in range(available_customers.shape[0]):
        cust = available_customers[i]
        this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # Best cost from other trucks
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
        if regret > best_regret or (regret == best_regret and this_cost < best_this_cost):
            best_regret = regret
            best_customer_idx = i
            best_this_cost = this_cost
    return best_customer_idx