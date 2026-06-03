def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    other_depot_dists = []
    for pos in truck_positions:
        if not np.array_equal(pos, current_position):
            other_depot_dists.append(np.linalg.norm(depot_position - pos))
    max_other = max(other_depot_dists) if other_depot_dists else 0.0
    best_idx = None
    best_max_return = float('inf')
    best_finish = float('inf')
    for i, cust in enumerate(available_customers):
        dist_to_cust = np.linalg.norm(current_position - cust)
        cust_to_depot = np.linalg.norm(depot_position - cust)
        finish = dist_to_cust + cust_to_depot
        max_return = max(finish, max_other)
        if max_return < best_max_return or (max_return == best_max_return and finish < best_finish):
            best_max_return = max_return
            best_finish = finish
            best_idx = i
    return best_idx