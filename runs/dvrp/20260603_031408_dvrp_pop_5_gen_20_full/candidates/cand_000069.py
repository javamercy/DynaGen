def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    other_dists = []
    for pos in truck_positions:
        if not np.array_equal(pos, current_position):
            other_dists.append(np.linalg.norm(depot_position - pos))
    max_other = max(other_dists) if other_dists else 0.0
    best_idx = None
    best_score = -float('inf')
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
        finish = cust_to_truck + cust_to_depot
        excess = max(0.0, finish - max_other)
        if max_other > 1e-6:
            penalty_weight = 0.3 * (1 + excess / max_other)
            if penalty_weight > 0.8:
                penalty_weight = 0.8
        else:
            penalty_weight = 0.5
        score = cust_to_depot - cust_to_truck + 0.6 * min_ot - penalty_weight * excess
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx