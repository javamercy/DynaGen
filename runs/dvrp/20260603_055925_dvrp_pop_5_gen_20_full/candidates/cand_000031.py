def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    n_trucks = len(truck_positions)
    if n_trucks == 1:
        distances = np.linalg.norm(available_customers - current_position, axis=1)
        return int(np.argmin(distances))
    # Identify current truck index
    current_truck_idx = int(np.argmin(np.linalg.norm(truck_positions - current_position, axis=1)))
    best_regret = -float('inf')
    best_idx = None
    for i, cust in enumerate(available_customers):
        d_current = np.linalg.norm(current_position - cust)
        d_depot = np.linalg.norm(cust - depot_position)
        total_current = d_current + d_depot
        min_total_other = float('inf')
        for j, pos in enumerate(truck_positions):
            if j == current_truck_idx:
                continue
            d_other = np.linalg.norm(pos - cust)
            total_other = d_other + d_depot
            if total_other < min_total_other:
                min_total_other = total_other
        regret = min_total_other - total_current
        if regret > best_regret:
            best_regret = regret
            best_idx = i
    if best_regret > 0:
        return best_idx
    else:
        return None