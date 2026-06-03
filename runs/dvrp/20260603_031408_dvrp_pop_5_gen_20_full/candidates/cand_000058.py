def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    best_idx = None
    best_score = -float('inf')
    for i, cust in enumerate(available_customers):
        cust_to_depot = np.linalg.norm(depot_position - cust)
        cust_to_truck = np.linalg.norm(current_position - cust)
        # max distance to any other truck (excluding current)
        max_ot = -float('inf')
        for pos in truck_positions:
            if not np.array_equal(pos, current_position):
                d = np.linalg.norm(pos - cust)
                if d > max_ot:
                    max_ot = d
        if max_ot == -float('inf'):
            max_ot = 0.0
        score = cust_to_depot - cust_to_truck + max_ot
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx