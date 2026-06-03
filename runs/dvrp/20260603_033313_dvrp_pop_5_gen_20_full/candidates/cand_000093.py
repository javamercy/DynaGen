import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    n_available = len(available_customers)
    n_trucks = len(truck_positions)
    
    # Find index of current truck
    current_truck_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    
    # Distances to depot for all trucks
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_dist = dist_to_depot[current_truck_idx]
    other_dist = np.delete(dist_to_depot, current_truck_idx)
    avg_other_dist = np.mean(other_dist) if len(other_dist) > 0 else 0.0
    
    # Urgency factor: higher when current truck is far relative to others
    urgency_factor = 1.0 + (current_dist / (avg_other_dist + 1e-8))
    
    best_regret = float('inf')
    best_idx = None
    best_cost_now = float('inf')
    
    for i, cust in enumerate(available_customers):
        cost_now = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        all_costs = [np.linalg.norm(truck - cust) + np.linalg.norm(cust - depot_position) for truck in truck_positions]
        sorted_costs = sorted(all_costs)
        current_cost = all_costs[current_truck_idx]
        # Find min and max alternative (excluding current truck)
        min_alt = None
        max_alt = None
        for c in sorted_costs:
            if np.isclose(c, current_cost, atol=1e-8):
                continue
            if min_alt is None:
                min_alt = c
            max_alt = c
        if min_alt is None:
            min_alt = current_cost
            max_alt = current_cost
        penalty_factor = 1.0 + n_available / (n_available + 10.0)
        regret = (cost_now - min_alt) + penalty_factor * max(0, cost_now - max_alt)
        regret_weighted = regret * urgency_factor
        if regret_weighted < best_regret or (regret_weighted == best_regret and cost_now < best_cost_now):
            best_regret = regret_weighted
            best_idx = i
            best_cost_now = cost_now
    
    # Dynamic wait threshold: decreases with urgency (truck far from depot -> smaller threshold)
    wait_threshold = 0.1 * (1.0 + n_available / 10.0) * (avg_other_dist / (current_dist + 1e-8))
    if best_regret > 0 and best_regret > wait_threshold * best_cost_now:
        return None
    return best_idx