def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # Find index of current truck in truck_positions
    current_truck_idx = None
    for idx, pos in enumerate(truck_positions):
        if np.array_equal(pos, current_position):
            current_truck_idx = idx
            break
    if current_truck_idx is None:
        # Fallback: shouldn't happen; use current_position as reference
        current_truck_idx = -1
    best_idx = None
    best_score = -float('inf')
    for i, cust in enumerate(available_customers):
        d_depot = np.linalg.norm(depot_position - cust)
        # Distance to nearest other truck (excluding current)
        d_other = min(
            np.linalg.norm(truck_positions[j] - cust)
            for j in range(len(truck_positions))
            if j != current_truck_idx
        )
        score = d_depot - d_other
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx