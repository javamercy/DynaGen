def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # Compute other trucks' direct return distances to depot
    other_dists = []
    for pos in truck_positions:
        if not np.array_equal(pos, current_position):
            other_dists.append(np.linalg.norm(depot_position - pos))
    other_max = max(other_dists) if other_dists else 0.0
    best_idx = None
    best_max = float('inf')
    best_isolation = -float('inf')
    for i, cust in enumerate(available_customers):
        active_est = np.linalg.norm(current_position - cust) + np.linalg.norm(depot_position - cust)
        cand_max = max(active_est, other_max)
        # compute isolation: min distance to other trucks
        isolation = float('inf')
        for pos in truck_positions:
            if not np.array_equal(pos, current_position):
                d = np.linalg.norm(pos - cust)
                if d < isolation:
                    isolation = d
        if isolation == float('inf'):
            isolation = 0.0
        if cand_max < best_max or (cand_max == best_max and isolation > best_isolation):
            best_max = cand_max
            best_isolation = isolation
            best_idx = i
    return best_idx