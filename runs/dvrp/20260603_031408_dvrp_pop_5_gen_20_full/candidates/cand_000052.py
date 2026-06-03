def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    best_idx = None
    best_score = -float('inf')
    best_time = None
    for i, cust in enumerate(available_customers):
        cust_to_depot = np.linalg.norm(depot_position - cust)
        cust_to_truck = np.linalg.norm(current_position - cust)
        min_ot = float('inf')
        for pos in truck_positions:
            if not np.array_equal(pos, current_position):
                d = np.linalg.norm(pos - cust)
                if d < min_ot:
                    min_ot = d
        if min_ot == float('inf'):
            min_ot = 0.0
        score = cust_to_depot - cust_to_truck + 0.5 * min_ot
        candidate_time = cust_to_truck + cust_to_depot
        if score > best_score:
            best_score = score
            best_idx = i
            best_time = candidate_time
    # compute max distance to depot among other trucks
    max_other_dist = 0.0
    for pos in truck_positions:
        if not np.array_equal(pos, current_position):
            d = np.linalg.norm(pos - depot_position)
            if d > max_other_dist:
                max_other_dist = d
    if best_time is not None and best_time > max_other_dist:
        return None
    return best_idx