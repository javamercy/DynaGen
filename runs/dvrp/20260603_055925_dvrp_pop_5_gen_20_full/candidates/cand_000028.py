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
        # Only one truck: go to nearest customer, but with depot incentive if few remain
        if len(available_customers) <= 1:
            # directly compute combined distance
            combined = np.linalg.norm(available_customers - current_position, axis=1) + \
                        np.linalg.norm(available_customers - depot_position, axis=1)
            return int(np.argmin(combined))
        else:
            distances = np.linalg.norm(available_customers - current_position, axis=1)
            return int(np.argmin(distances))
    
    # Find index of current truck
    current_truck_idx = int(np.argmin(np.linalg.norm(truck_positions - current_position, axis=1)))
    
    # Precompute distances from current truck to all customers
    current_dists = np.linalg.norm(available_customers - current_position, axis=1)
    
    # For each customer, find distance to nearest other truck
    other_dists = []
    for i, cust in enumerate(available_customers):
        dists_to_other_trucks = np.linalg.norm(truck_positions - cust, axis=1)
        # exclude current truck's distance
        dists_to_other_trucks = np.delete(dists_to_other_trucks, current_truck_idx)
        min_other = np.min(dists_to_other_trucks) if len(dists_to_other_trucks) > 0 else float('inf')
        other_dists.append(min_other)
    other_dists = np.array(other_dists)
    
    # Compute depot distances from customers
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    
    if len(available_customers) <= n_trucks:
        # Add depot incentive: penalize customers far from depot
        regrets = other_dists - current_dists - 0.5 * cust_to_depot
    else:
        regrets = other_dists - current_dists
    
    max_regret = np.max(regrets)
    if max_regret >= 0:
        return int(np.argmax(regrets))
    else:
        return None