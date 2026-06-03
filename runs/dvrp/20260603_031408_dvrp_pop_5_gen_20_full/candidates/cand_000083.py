def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    other_dist_to_depot = []
    for pos in truck_positions:
        if not np.array_equal(pos, current_position):
            other_dist_to_depot.append(np.linalg.norm(depot_position - pos))
    max_other = max(other_dist_to_depot) if other_dist_to_depot else 0.0
    best_idx = None
    best_score = float('inf')
    best_finish = float('inf')
    for i, cust in enumerate(available_customers):
        cust_to_truck = np.linalg.norm(current_position - cust)
        cust_to_depot = np.linalg.norm(depot_position - cust)
        finish = cust_to_truck + cust_to_depot
        new_max = max(finish, max_other)
        score = new_max
        if score < best_score or (score == best_score and finish < best_finish):
            best_score = score
            best_finish = finish
            best_idx = i
    return best_idx