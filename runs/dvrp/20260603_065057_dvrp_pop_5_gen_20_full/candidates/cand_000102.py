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
    dists = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(dists)
    best_score = np.inf  # we want to minimize max_this
    best_customer_idx = -1
    best_this_cost = np.inf
    for i in range(available_customers.shape[0]):
        cust = available_customers[i]
        this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # compute costs for other trucks
        other_costs = []
        for j in range(n_trucks):
            if j == current_idx:
                continue
            other_cost = np.linalg.norm(truck_positions[j] - cust) + np.linalg.norm(cust - depot_position)
            other_costs.append(other_cost)
        if len(other_costs) > 0:
            max_other = max(other_costs)
        else:
            max_other = -np.inf  # so max_this = this_cost
        max_this = max(this_cost, max_other)
        # we want to minimize max_this
        if max_this < best_score or (max_this == best_score and this_cost < best_this_cost):
            best_score = max_this
            best_customer_idx = i
            best_this_cost = this_cost
    return best_customer_idx