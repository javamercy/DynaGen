import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    n_trucks = len(truck_positions)
    if n_trucks == 1:
        distances = np.linalg.norm(available_customers - current_position, axis=1)
        return int(np.argmin(distances))
    # Find current truck index
    current_truck_idx = int(np.argmin(np.linalg.norm(truck_positions - current_position, axis=1)))
    # Compute depot distances for all trucks
    truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    max_depot_dist = np.max(truck_depot_dists)
    if max_depot_dist == 0:
        weights = np.ones(n_trucks)
    else:
        weights = 1.0 + truck_depot_dists / max_depot_dist  # between 1 and 2
    best_regret = -float('inf')
    best_idx = None
    best_secondary = -float('inf')
    for i, cust in enumerate(available_customers):
        d_curr = np.linalg.norm(current_position - cust)
        cust_depot = np.linalg.norm(cust - depot_position)
        current_cost = d_curr + weights[current_truck_idx] * cust_depot
        other_costs = []
        for j, pos in enumerate(truck_positions):
            if j == current_truck_idx:
                continue
            d_other = np.linalg.norm(pos - cust)
            other_cost = d_other + weights[j] * cust_depot
            other_costs.append(other_cost)
        best_other = min(other_costs) if other_costs else float('inf')
        regret = best_other - current_cost
        secondary = -d_curr  # tie-breaker: prefer closer customers
        if (regret > best_regret) or (regret == best_regret and secondary > best_secondary):
            best_regret = regret
            best_idx = i
            best_secondary = secondary
    if best_regret >= 0:
        return best_idx
    else:
        return None