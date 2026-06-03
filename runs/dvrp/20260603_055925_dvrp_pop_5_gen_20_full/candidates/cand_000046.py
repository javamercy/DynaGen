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
    other_dists = []
    for i, cust in enumerate(available_customers):
        dists_to_other_trucks = np.linalg.norm(truck_positions - cust, axis=1)
        dists_to_other_trucks = np.delete(dists_to_other_trucks, current_truck_idx)
        min_other = np.min(dists_to_other_trucks) if len(dists_to_other_trucks) > 0 else float('inf')
        other_dists.append(min_other)
    other_dists = np.array(other_dists)
    regrets = other_dists - current_dists
    if len(available_customers) > n_trucks:
        max_regret = np.max(regrets)
        if max_regret >= 0:
            return int(np.argmax(regrets))
        else:
            return None
    else:
        # Few customers: incorporate depot benefit
        customer_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
        current_to_depot = np.linalg.norm(current_position - depot_position)
        depot_benefit = current_to_depot - (current_dists + customer_to_depot)
        scores = regrets + 0.5 * depot_benefit
        max_score = np.max(scores)
        if max_score >= 0:
            return int(np.argmax(scores))
        else:
            return None