import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    n_trucks = len(truck_positions)
    # Special case: only one truck
    if n_trucks == 1:
        best_idx = None
        best_cost = float('inf')
        for i, cust in enumerate(available_customers):
            cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
            if cost < best_cost:
                best_cost = cost
                best_idx = i
        return best_idx
    
    best_eff_regret = float('inf')
    best_idx = None
    best_cost_now = float('inf')
    
    # Identify current truck index by comparing positions
    current_truck_idx = None
    for j in range(n_trucks):
        if np.allclose(truck_positions[j], current_position):
            current_truck_idx = j
            break
    if current_truck_idx is None:
        # Fallback: assume first? Should not happen; but handle gracefully
        current_truck_idx = 0
    
    for i, cust in enumerate(available_customers):
        cost_now = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        alt_costs = []
        other_direct = []
        for j in range(n_trucks):
            if j == current_truck_idx:
                continue
            truck = truck_positions[j]
            alt_costs.append(np.linalg.norm(truck - cust) + np.linalg.norm(cust - depot_position))
            other_direct.append(np.linalg.norm(truck - depot_position))
        min_alt = min(alt_costs) if alt_costs else float('inf')
        regret = cost_now - min_alt
        max_other_return = max(other_direct) if other_direct else 0.0
        excess = max(0.0, cost_now - max_other_return)
        eff_regret = regret + excess
        if eff_regret < best_eff_regret or (eff_regret == best_eff_regret and cost_now < best_cost_now):
            best_eff_regret = eff_regret
            best_idx = i
            best_cost_now = cost_now
    
    # Wait condition: at depot and best effective regret > 0
    at_depot = np.linalg.norm(current_position - depot_position) < 1e-6
    if at_depot and best_eff_regret > 0:
        return None
    return best_idx