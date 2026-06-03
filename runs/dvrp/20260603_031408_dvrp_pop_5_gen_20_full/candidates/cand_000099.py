def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # compute current max other truck distance to depot
    other_dists = []
    for pos in truck_positions:
        if not np.array_equal(pos, current_position):
            other_dists.append(np.linalg.norm(depot_position - pos))
    max_other = max(other_dists) if other_dists else 0.0
    # compute adaptive penalty weight based on truck's own distance to depot vs average other
    my_dist_to_depot = np.linalg.norm(current_position - depot_position)
    avg_other_dist = np.mean(other_dists) if other_dists else my_dist_to_depot
    pressure = my_dist_to_depot / (avg_other_dist + 1e-8)
    penalty_weight = 0.5 + 0.5 * min(pressure, 2.0)  # range [0.5, 1.5]
    isolation_weight = 0.5
    best_idx = None
    best_score = -float('inf')
    for i, cust in enumerate(available_customers):
        cust_to_depot = np.linalg.norm(depot_position - cust)
        cust_to_truck = np.linalg.norm(current_position - cust)
        # isolation: distance to nearest other truck
        min_ot = float('inf')
        for pos in truck_positions:
            if not np.array_equal(pos, current_position):
                d = np.linalg.norm(pos - cust)
                if d < min_ot:
                    min_ot = d
        if min_ot == float('inf'):
            min_ot = 0.0
        # depot-return pressure: estimated travel if go to cust then depot
        finish = cust_to_truck + cust_to_depot
        penalty = max(0.0, finish - max_other)
        score = cust_to_depot - cust_to_truck + isolation_weight * min_ot - penalty_weight * penalty
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx