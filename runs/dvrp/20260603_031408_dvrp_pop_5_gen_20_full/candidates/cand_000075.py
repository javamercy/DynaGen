def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # compute max distance from depot among other trucks
    other_dists = []
    for i, pos in enumerate(truck_positions):
        if not np.array_equal(pos, current_position):
            other_dists.append(np.linalg.norm(depot_position - pos))
    max_other = max(other_dists) if other_dists else 0.0
    best_idx = None
    best_score = -float('inf')
    for i, cust in enumerate(available_customers):
        cust_to_truck = np.linalg.norm(current_position - cust)
        cust_to_depot = np.linalg.norm(depot_position - cust)
        # isolation: distance to nearest other truck
        min_ot = float('inf')
        for j, pos in enumerate(truck_positions):
            if not np.array_equal(pos, current_position):
                d = np.linalg.norm(pos - cust)
                if d < min_ot:
                    min_ot = d
        if min_ot == float('inf'):
            min_ot = 0.0
        finish = cust_to_truck + cust_to_depot
        penalty = max(0.0, finish - max_other)
        score = min_ot - 0.5 * penalty
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx