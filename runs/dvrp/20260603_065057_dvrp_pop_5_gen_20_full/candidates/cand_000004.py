import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    # Precompute distances
    n_trucks = truck_positions.shape[0]
    # Find index of current truck in truck_positions (assume it's there? we need to exclude it)
    # Since we don't know which index corresponds to current_position, we compute distance to current_position and find the closest truck (should be 0 distance)
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diff)
    # Compute for each customer
    best_regret = -np.inf
    best_customer_idx = -1
    best_this_cost = np.inf
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
        # If no other trucks, other_costs is empty; then no regret
        if len(other_costs) > 0:
            other_min = min(other_costs)
            regret = max(0, other_min - this_cost)
        else:
            regret = 0
        # Choose based on regret, tie-break by this_cost (lower better)
        if regret > best_regret or (regret == best_regret and this_cost < best_this_cost):
            best_regret = regret
            best_customer_idx = i
            best_this_cost = this_cost
    return best_customer_idx