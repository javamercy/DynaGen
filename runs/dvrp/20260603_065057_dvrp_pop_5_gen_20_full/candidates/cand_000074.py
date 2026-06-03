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
    # Identify current truck index
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diff)
    # Compute other trucks' distances to depot
    other_depot_dists = []
    for j in range(n_trucks):
        if j != current_idx:
            other_depot_dists.append(np.linalg.norm(truck_positions[j] - depot_position))
    max_other_dist = max(other_depot_dists) if other_depot_dists else -np.inf
    # Evaluate each customer
    best_key = None
    best_idx = -1
    for i in range(available_customers.shape[0]):
        cust = available_customers[i]
        this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        if n_trucks == 1:
            M_i = this_cost
            regret = 0.0
        else:
            # compute best other cost
            other_costs = []
            for j in range(n_trucks):
                if j == current_idx:
                    continue
                other_cost = np.linalg.norm(truck_positions[j] - cust) + np.linalg.norm(cust - depot_position)
                other_costs.append(other_cost)
            best_other = min(other_costs)
            regret = max(0.0, best_other - this_cost)
            M_i = max(this_cost, max_other_dist)
        # Key: primary - M_i, secondary -regret, tertiary this_cost
        key = (M_i, -regret, this_cost)
        if best_key is None or key < best_key:
            best_key = key
            best_idx = i
    return best_idx