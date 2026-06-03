import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    # Find current truck index
    dist_to_trucks = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(dist_to_trucks)
    n_trucks = truck_positions.shape[0]
    # Precompute current depot distances for all trucks
    truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_depot_dist = truck_depot_dists[current_idx]
    max_other_depot = np.max(np.delete(truck_depot_dists, current_idx)) if n_trucks > 1 else 0.0
    current_max_before = max(current_depot_dist, max_other_depot)
    
    best_score = -np.inf
    best_customer_idx = -1
    best_this_cost = np.inf
    for i in range(available_customers.shape[0]):
        cust = available_customers[i]
        this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # Other trucks' cost to serve this customer
        other_costs = []
        for j in range(n_trucks):
            if j == current_idx:
                continue
            other_cost = np.linalg.norm(truck_positions[j] - cust) + np.linalg.norm(cust - depot_position)
            other_costs.append(other_cost)
        other_min = min(other_costs) if other_costs else 0.0
        regret = max(0.0, other_min - this_cost)
        # Increase in max estimated return time
        new_this_time = this_cost
        new_max = max(new_this_time, max_other_depot)
        increase = new_max - current_max_before
        # Score: regret minus penalty for increase
        w = 1.0  # weight, could be tuned
        score = regret - w * increase
        if score > best_score or (score == best_score and this_cost < best_this_cost):
            best_score = score
            best_customer_idx = i
            best_this_cost = this_cost
    return best_customer_idx