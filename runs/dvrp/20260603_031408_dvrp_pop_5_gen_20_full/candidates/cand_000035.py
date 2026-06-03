def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # Compute distances of all trucks to depot
    all_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_depot_dist = np.linalg.norm(current_position - depot_position)
    # Find max other depot distance (excluding current truck)
    max_all = np.max(all_depot_dists)
    if np.isclose(max_all, current_depot_dist):
        sorted_dists = np.sort(all_depot_dists)
        if len(sorted_dists) >= 2:
            max_other = sorted_dists[-2]
        else:
            max_other = current_depot_dist
    else:
        max_other = max_all
    best_idx = None
    best_cost = None
    for i, cust in enumerate(available_customers):
        d_truck2cust = np.linalg.norm(current_position - cust)
        d_cust2depot = np.linalg.norm(cust - depot_position)
        active_total = d_truck2cust + d_cust2depot
        makespan = max(active_total, max_other)
        # Primary: makespan, secondary: active_total (smaller better)
        cost = (makespan, active_total)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_idx = i
    return best_idx