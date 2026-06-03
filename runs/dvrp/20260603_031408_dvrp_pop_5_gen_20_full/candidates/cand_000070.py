def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # compute current max distance of other trucks to depot
    max_other = 0.0
    for pos in truck_positions:
        if not np.array_equal(pos, current_position):
            d = np.linalg.norm(depot_position - pos)
            if d > max_other:
                max_other = d
    best_idx = None
    best_val = float('inf')
    best_travel = float('inf')
    for i, cust in enumerate(available_customers):
        travel = np.linalg.norm(current_position - cust)
        return_dist = np.linalg.norm(cust - depot_position)
        my_total = travel + return_dist
        max_val = max(my_total, max_other)
        # tie-break: prefer shorter travel
        if max_val < best_val or (max_val == best_val and travel < best_travel):
            best_val = max_val
            best_travel = travel
            best_idx = i
    return best_idx