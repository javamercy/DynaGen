import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    n_available = len(available_customers)
    n_trucks = len(truck_positions)
    # Identify current truck index by closest position
    current_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    # Compute base distances to depot for all trucks (current positions, not after service)
    base_dist = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_base = base_dist[current_idx]
    avg_base = np.mean(base_dist)
    # depot_urgency factor: if current truck is further than average, it's more urgent
    depot_urgency = current_base / (avg_base + 1e-6)
    
    best_regret = float('inf')
    best_idx = None
    best_cost_now = float('inf')
    for i, cust in enumerate(available_customers):
        dist_to_cust = np.linalg.norm(current_position - cust)
        dist_cust_to_depot = np.linalg.norm(cust - depot_position)
        cost_now = dist_to_cust + dist_cust_to_depot
        # Compute costs for all trucks
        all_costs = [np.linalg.norm(truck_positions[j] - cust) + dist_cust_to_depot for j in range(n_trucks)]
        sorted_costs = sorted(all_costs)
        # min alternative (excluding current truck)
        if n_trucks > 1:
            if np.isclose(sorted_costs[0], cost_now, atol=1e-8):
                min_alt = sorted_costs[1]
            else:
                min_alt = sorted_costs[0]
            # max alternative (excluding current truck)
            if np.isclose(sorted_costs[-1], cost_now, atol=1e-8):
                max_alt = sorted_costs[-2]
            else:
                max_alt = sorted_costs[-1]
        else:
            min_alt = cost_now
            max_alt = cost_now
        # Adaptive penalty: base 1.0 + depot_urgency
        penalty_factor = 1.0 + depot_urgency
        regret = (cost_now - min_alt) + penalty_factor * max(0, cost_now - max_alt)
        if regret < best_regret or (np.isclose(regret, best_regret) and cost_now < best_cost_now):
            best_regret = regret
            best_idx = i
            best_cost_now = cost_now
    # Wait threshold: increases with available customers, but decreases with depot_urgency (far truck waits less)
    wait_threshold = 0.1 * (1.0 + n_available / 10.0) / (depot_urgency + 0.5)  # offset to avoid division by zero
    if best_regret > 0 and best_regret > wait_threshold * best_cost_now:
        return None
    return best_idx