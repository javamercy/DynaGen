import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    
    best_regret = None
    best_idx = None
    best_cost = None
    
    for i, cust in enumerate(available_customers):
        cost_me = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # Compute costs for all trucks (including current) to this customer
        all_costs = [np.linalg.norm(truck - cust) + np.linalg.norm(cust - depot_position) for truck in truck_positions]
        sorted_costs = sorted(all_costs)
        # Find min alternative (excluding current truck)
        if len(sorted_costs) > 1 and np.isclose(sorted_costs[0], cost_me, atol=1e-8):
            min_alt = sorted_costs[1]
        else:
            min_alt = sorted_costs[0]
        # Find max alternative (excluding current truck)
        if len(sorted_costs) > 1 and np.isclose(sorted_costs[-1], cost_me, atol=1e-8):
            max_alt = sorted_costs[-2]
        else:
            max_alt = sorted_costs[-1]
        
        regret = (cost_me - min_alt) + 1.0 * max(0, cost_me - max_alt)
        
        if best_regret is None or regret < best_regret or (regret == best_regret and cost_me < best_cost):
            best_regret = regret
            best_idx = i
            best_cost = cost_me
    
    # Wait if all customers have positive regret
    if best_regret > 0:
        return None
    return best_idx