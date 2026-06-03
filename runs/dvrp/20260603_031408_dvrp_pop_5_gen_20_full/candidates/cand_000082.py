def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    current_dist = np.linalg.norm(depot_position - current_position)
    other_dists = []
    for pos in truck_positions:
        if not np.array_equal(pos, current_position):
            other_dists.append(np.linalg.norm(depot_position - pos))
    max_other = max(other_dists) if other_dists else 0.0
    best_idx = None
    best_ttt = float('inf')
    best_finish = float('inf')
    best_isolation = -float('inf')
    for i, cust in enumerate(available_customers):
        cust_to_truck = np.linalg.norm(current_position - cust)
        cust_to_depot = np.linalg.norm(depot_position - cust)
        finish = cust_to_truck + cust_to_depot
        ttt = max(finish, max_other)
        # isolation: min distance to other trucks
        min_ot = float('inf')
        for pos in truck_positions:
            if not np.array_equal(pos, current_position):
                d = np.linalg.norm(pos - cust)
                if d < min_ot:
                    min_ot = d
        if min_ot == float('inf'):
            min_ot = 0.0
        # primary: minimize ttt; secondary: smaller finish; tertiary: larger isolation
        if (ttt < best_ttt) or (ttt == best_ttt and finish < best_finish) or (ttt == best_ttt and finish == best_finish and min_ot > best_isolation):
            best_ttt = ttt
            best_finish = finish
            best_isolation = min_ot
            best_idx = i
    return best_idx