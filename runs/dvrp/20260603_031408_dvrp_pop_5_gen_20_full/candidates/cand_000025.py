def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # Max distance from other trucks to depot (lower bound)
    other_returns = [
        np.linalg.norm(pos - depot_position)
        for pos in truck_positions
        if not np.allclose(pos, current_position)
    ]
    max_other_return = max(other_returns, default=0.0)
    best_idx = None
    best_ttt = float('inf')
    best_dist = float('inf')
    for i, customer in enumerate(available_customers):
        first_dist = np.linalg.norm(customer - current_position)
        return_dist = np.linalg.norm(customer - depot_position)
        active_return = first_dist + return_dist
        ttt = max(active_return, max_other_return)
        if ttt < best_ttt or (ttt == best_ttt and first_dist < best_dist):
            best_ttt = ttt
            best_idx = i
            best_dist = first_dist
    return best_idx