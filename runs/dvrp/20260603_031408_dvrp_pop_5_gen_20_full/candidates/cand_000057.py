def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    # identify active truck index
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    # distances from each truck to depot
    truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    if len(truck_positions) > 1:
        other_max_depot = np.max(np.delete(truck_depot_dists, active_idx))
    else:
        other_max_depot = 0.0

    best_idx = None
    best_cost = float('inf')

    for i, cust in enumerate(available_customers):
        d_truck = np.linalg.norm(current_position - cust)
        d_depot = np.linalg.norm(cust - depot_position)
        # distance to other trucks
        dists_to_all = np.linalg.norm(truck_positions - cust, axis=1)
        dists_to_all[active_idx] = float('inf')  # exclude self
        min_other_dist = np.min(dists_to_all)
        # penalty for making active truck's return distance exceed other max
        my_return = d_truck + d_depot
        penalty = max(0, my_return - other_max_depot) * 0.2  # weight
        cost = d_truck - 0.5 * d_depot - 0.3 * min_other_dist + penalty
        if cost < best_cost:
            best_cost = cost
            best_idx = i

    return best_idx