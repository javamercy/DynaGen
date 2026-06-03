import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # Compute current truck's direct return distance
    current_direct = np.linalg.norm(current_position - depot_position)
    # Compute other trucks' direct return distances
    other_directs = []
    for truck in truck_positions:
        if not np.allclose(truck, current_position):
            other_directs.append(np.linalg.norm(truck - depot_position))
    avg_other_direct = np.mean(other_directs) if other_directs else 0.0
    # Dynamic penalty weight based on relative direct return
    if avg_other_direct > 1e-9:
        weight = current_direct / avg_other_direct
        # Clamp to avoid extreme values
        weight = max(0.5, min(3.0, weight))
    else:
        weight = 1.5  # default from parent
    best_regret = float('inf')
    best_idx = None
    best_cost = float('inf')
    for i, cust in enumerate(available_customers):
        curr_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        all_costs = [np.linalg.norm(truck - cust) + np.linalg.norm(cust - depot_position) for truck in truck_positions]
        sorted_costs = sorted(all_costs)
        min_alt = sorted_costs[0]
        if len(sorted_costs) > 1 and np.isclose(min_alt, curr_cost, atol=1e-8):
            min_alt = sorted_costs[1]
        max_alt = sorted_costs[-1]
        if len(sorted_costs) > 1 and np.isclose(max_alt, curr_cost, atol=1e-8):
            max_alt = sorted_costs[-2]
        regret = (curr_cost - min_alt) + weight * max(0, curr_cost - max_alt)
        if regret < best_regret or (np.isclose(regret, best_regret) and curr_cost < best_cost):
            best_regret = regret
            best_idx = i
            best_cost = curr_cost
    at_depot = np.linalg.norm(current_position - depot_position) < 1e-6
    if best_regret > 0 and at_depot and current_direct < avg_other_direct:
        return None
    return best_idx