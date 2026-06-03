def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # Find index of current truck in truck_positions
    diffs = truck_positions - current_position
    current_idx = np.argmin(np.linalg.norm(diffs, axis=1))
    # Precompute distances from depot and to current truck
    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)
    truck_dists = np.linalg.norm(available_customers - current_position, axis=1)
    # Compute min distance to other trucks for each customer
    other_dists = []
    for cust in available_customers:
        dists_to_all = np.linalg.norm(truck_positions - cust, axis=1)
        # Set distance to current truck to large value
        dists_to_all[current_idx] = np.inf
        min_other = np.min(dists_to_all)
        other_dists.append(min_other)
    other_dists = np.array(other_dists)
    # Cost = depot_dist - truck_dist - min_other_dist (alpha=1)
    costs = depot_dists - truck_dists - other_dists
    best_idx = np.argmax(costs)
    return int(best_idx)