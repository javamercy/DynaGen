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
    
    current_truck_idx = int(np.argmin(np.linalg.norm(truck_positions - current_position, axis=1)))
    current_dists = np.linalg.norm(available_customers - current_position, axis=1)
    
    other_dists = np.full(len(available_customers), np.inf)
    for i, cust in enumerate(available_customers):
        dists_to_all = np.linalg.norm(truck_positions - cust, axis=1)
        dists_to_all[current_truck_idx] = np.inf
        other_dists[i] = np.min(dists_to_all)
    
    regrets = other_dists - current_dists
    
    remaining = len(available_customers)
    if remaining <= 5:
        depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)
        w = 0.2
        regrets = regrets - w * depot_dists
    
    max_regret = np.max(regrets)
    if max_regret >= 0:
        return int(np.argmax(regrets))
    else:
        return None