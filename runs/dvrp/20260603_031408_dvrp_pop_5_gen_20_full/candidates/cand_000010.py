def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    best_idx = None
    best_cost = -float('inf')
    for i, cust in enumerate(available_customers):
        dist_to_truck = np.linalg.norm(current_position - cust)
        dist_to_depot = np.linalg.norm(depot_position - cust)
        cost = dist_to_depot - dist_to_truck
        if cost > best_cost:
            best_cost = cost
            best_idx = i
    return best_idx