def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # compute max distance to depot among other trucks
    other_max = 0.0
    for pos in truck_positions:
        if not np.array_equal(pos, current_position):
            d = np.linalg.norm(depot_position - pos)
            if d > other_max:
                other_max = d
    best_idx = None
    best_score = -float('inf')
    for i, cust in enumerate(available_customers):
        finish = np.linalg.norm(current_position - cust) + np.linalg.norm(depot_position - cust)
        tentative_max = max(finish, other_max)
        # small isolation bonus
        isolation = min([np.linalg.norm(pos - cust) for pos in truck_positions if not np.array_equal(pos, current_position)], default=0.0)
        score = -tentative_max + 0.1 * isolation
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx