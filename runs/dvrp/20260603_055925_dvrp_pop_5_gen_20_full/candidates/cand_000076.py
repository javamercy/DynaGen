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

    # Distances from current truck to customers
    current_dists = np.linalg.norm(available_customers - current_position, axis=1)

    # For each customer, distance to nearest other truck
    other_dists = []
    for i, cust in enumerate(available_customers):
        dists_to_other = np.linalg.norm(truck_positions - cust, axis=1)
        dists_to_other = np.delete(dists_to_other, current_truck_idx)
        min_other = np.min(dists_to_other) if len(dists_to_other) > 0 else float('inf')
        other_dists.append(min_other)
    other_dists = np.array(other_dists)

    # Base regret
    regrets = other_dists - current_dists

    # Depot incentive
    n_rem = len(available_customers)
    current_to_depot = np.linalg.norm(current_position - depot_position)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    if n_rem <= 3:
        alpha = 0.3
    else:
        alpha = 0.05
    depot_bonus = (current_to_depot - cust_to_depot) * alpha
    adjusted_regret = regrets + depot_bonus

    max_adjusted = np.max(adjusted_regret)
    if max_adjusted >= 0:
        return int(np.argmax(adjusted_regret))
    else:
        # If regret only slightly negative, still serve to avoid long waits
        threshold = -0.5  # tune
        best_idx = int(np.argmax(adjusted_regret))
        if adjusted_regret[best_idx] > threshold:
            return best_idx
        else:
            return None