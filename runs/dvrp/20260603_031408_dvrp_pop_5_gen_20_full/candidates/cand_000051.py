def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    other_dists = [np.linalg.norm(depot - pos) for i, pos in enumerate(truck_positions) if not np.array_equal(pos, current_position)]
    max_other = max(other_dists) if other_dists else 0.0
    best_idx = None
    best_makespan = float('inf')
    best_cust_depot = -float('inf')
    for i, cust in enumerate(available_customers):
        truck_return = np.linalg.norm(current_position - cust) + np.linalg.norm(depot - cust)
        makespan = max(truck_return, max_other)
        if makespan < best_makespan or (makespan == best_makespan and np.linalg.norm(depot - cust) > best_cust_depot):
            best_makespan = makespan
            best_cust_depot = np.linalg.norm(depot - cust)
            best_idx = i
    return best_idx