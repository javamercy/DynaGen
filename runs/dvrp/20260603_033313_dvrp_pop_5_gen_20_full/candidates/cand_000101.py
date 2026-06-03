import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    
    best_regret = float('inf')
    best_idx = None
    best_cost_now = None
    best_min_alt = None
    best_std = None
    
    # Depot bias factor for return distance
    alpha_ret = 1.1
    
    for i, cust in enumerate(available_customers):
        # Cost with depot bias: distance to customer + alpha_ret * distance customer to depot
        cost_now = np.linalg.norm(current_position - cust) + alpha_ret * np.linalg.norm(cust - depot_position)
        
        # Compute alternative costs for all trucks (including current truck? Actually all trucks positions)
        all_costs = [np.linalg.norm(truck - cust) + alpha_ret * np.linalg.norm(cust - depot_position) for truck in truck_positions]
        sorted_costs = sorted(all_costs)
        
        # Find min alternative cost (exclude current truck's cost if it is the minimum)
        min_alt = sorted_costs[0]
        if np.isclose(min_alt, cost_now, atol=1e-8) and len(sorted_costs) > 1:
            min_alt = sorted_costs[1]
        
        # Find max alternative cost (exclude current truck's cost if it is the maximum)
        max_alt = sorted_costs[-1]
        if np.isclose(max_alt, cost_now, atol=1e-8) and len(sorted_costs) > 1:
            max_alt = sorted_costs[-2]
        
        # Adaptive penalty factor based on ratio of min to max
        if max_alt > 0:
            penalty = 1.0 + 0.5 * (min_alt / max_alt)
        else:
            penalty = 1.5
        
        regret = (cost_now - min_alt) + penalty * max(0, cost_now - max_alt)
        
        if regret < best_regret:
            best_regret = regret
            best_idx = i
            best_cost_now = cost_now
            best_min_alt = min_alt
            best_std = np.std(all_costs)
    
    # Adaptive wait threshold using coefficient of variation
    epsilon = 1e-6
    if best_min_alt > 0:
        cv = best_std / best_min_alt
        # Wait more when variation is high; alpha scales with cv
        alpha_wait = 0.1 * cv / (1.0 + cv)  # ranges 0 to 0.1
    else:
        alpha_wait = 0.05
    
    if best_regret > 0 and best_cost_now > (1.0 + alpha_wait) * best_min_alt:
        return None
    
    return best_idx