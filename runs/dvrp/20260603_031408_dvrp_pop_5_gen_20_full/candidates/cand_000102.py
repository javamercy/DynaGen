def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # compute current max other truck distance to depot
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
        finish = cust_to_truck + cust_to_depot
        future_max = max(max_other, finish)
        # isolation: distance to nearest other truck
        min_ot = float('inf')
        for pos in truck_positions:
            if not np.array_equal(pos, current_position):
                d = np.linalg.norm(pos - cust)
                if d < min_ot:
                    min_ot = d
        if min_ot == float('inf'):
            min_ot = 0.0
        # score: minimize future_max, with bonus for isolation and slight depo-return incentive
        score = -future_max + 0.7 * min_ot + 0.1 * (cust_to_depot - cust_to_truck)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx