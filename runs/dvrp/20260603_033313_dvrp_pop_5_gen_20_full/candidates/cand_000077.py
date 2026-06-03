import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    
    n_trucks = len(truck_positions)
    n_cust = len(available_customers)
    
    # find active truck index
    active_idx = np.where(np.all(truck_positions == current_position, axis=1))[0][0]
    
    # distances from each customer to depot
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)  # (n_cust,)
    
    # distances from each truck to each customer: (n_cust, n_trucks)
    truck_to_cust = np.linalg.norm(
        available_customers[:, np.newaxis, :] - truck_positions[np.newaxis, :, :], axis=2
    )
    
    # cost matrix: truck_to_cust + cust_to_depot (broadcasted)
    costs = truck_to_cust + cust_to_depot[:, np.newaxis]  # (n_cust, n_trucks)
    
    # penalized cost for active truck
    alpha = 1.0
    active_depot_dist = np.linalg.norm(current_position - depot_position)
    active_penalty = alpha * active_depot_dist
    costs[:, active_idx] += active_penalty  # add penalty only to active truck's costs
    
    # compute best and second best costs for each customer
    sorted_costs = np.sort(costs, axis=1)
    best_cost = sorted_costs[:, 0]
    second_best = sorted_costs[:, 1]
    best_truck = np.argmin(costs, axis=1)  # (n_cust,), truck indices
    
    # regret = second_best - best_cost
    regrets = second_best - best_cost
    
    # mask: only consider customers where active truck is best
    valid_mask = (best_truck == active_idx)
    
    if np.any(valid_mask):
        # among valid, pick the one with highest regret
        best_idx = np.argmax(regrets * valid_mask)  # zeros for invalid have lower regret? but we want max only among valid
        # Since invalid have 0 regret? Actually if invalid, regret could be positive, but we want to ignore.
        # Use masked array or just multiply and then if regret zero for invalid, it's fine if valid gives positive.
        # Better: set invalid regrets to -inf
        masked_regrets = np.where(valid_mask, regrets, -np.inf)
        best_idx = np.argmax(masked_regrets)
        if masked_regrets[best_idx] != -np.inf:
            return int(best_idx)
    
    # if no valid customer, wait (return None)
    return None