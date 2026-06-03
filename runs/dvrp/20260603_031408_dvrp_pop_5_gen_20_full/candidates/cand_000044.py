def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    best_idx = None
    best_cost = -float('inf')
    n_trucks = len(truck_positions)
    for i, cust in enumerate(available_customers):
        dist_to_truck = np.linalg.norm(current_position - cust)
        dist_to_depot = np.linalg.norm(depot_position - cust)
        if n_trucks > 1:
            # distance to nearest other truck
            dists_other = [np.linalg.norm(truck_positions[j] - cust) for j in range(n_trucks) if not np.array_equal(truck_positions[j], current_position)]
            min_dist_other = min(dists_other) if dists_other else 0
        else:
            min_dist_other = 0
        cost = dist_to_depot - dist_to_truck - min_dist_other
        if cost > best_cost:
            best_cost = cost
            best_idx = i
    return best_idx