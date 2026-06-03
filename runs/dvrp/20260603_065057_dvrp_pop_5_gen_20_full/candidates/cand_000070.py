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
    # identify current truck index
    dist_to_trucks = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(dist_to_trucks)
    best_idx = None
    best_max = np.inf
    best_total = np.inf
    for i in range(available_customers.shape[0]):
        cust = available_customers[i]
        costs = np.zeros(n_trucks)
        for j in range(n_trucks):
            if j == current_idx:
                costs[j] = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
            else:
                costs[j] = np.linalg.norm(truck_positions[j] - depot_position)
        max_cost = np.max(costs)
        total_cost = np.sum(costs)
        if max_cost < best_max or (max_cost == best_max and total_cost < best_total):
            best_max = max_cost
            best_total = total_cost
            best_idx = i
    return best_idx