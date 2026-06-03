def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # compute current max other truck distance to depot (their return time if they go directly)
    other_dists = []
    for pos in truck_positions:
        if not np.array_equal(pos, current_position):
            other_dists.append(np.linalg.norm(depot_position - pos))
    max_other = max(other_dists) if other_dists else 0.0
    best_idx = None
    best_score = float('inf')
    for i, cust in enumerate(available_customers):
        own_finish = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # compute min distance from other trucks to this customer
        min_ot = float('inf')
        for pos in truck_positions:
            if not np.array_equal(pos, current_position):
                d = np.linalg.norm(pos - cust)
                if d < min_ot:
                    min_ot = d
        if min_ot == float('inf'):
            min_ot = 0.0
        other_finish_if_served = min_ot + np.linalg.norm(cust - depot_position)
        regret = own_finish - other_finish_if_served   # positive means current truck is worse
        new_max = max(own_finish, max_other)
        # score to minimize: combine new_max and regret (0.5 weight)
        score = new_max + 0.5 * regret
        if score < best_score:
            best_score = score
            best_idx = i
    return best_idx