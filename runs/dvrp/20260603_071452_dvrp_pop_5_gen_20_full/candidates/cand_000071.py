import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None

    n_trucks = len(truck_positions)
    # Find active truck index
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")

    depot_dist_active = np.linalg.norm(current_position - depot_position)
    # Distances to depot for other trucks
    other_depot_dists = []
    for j in range(n_trucks):
        if j != active_idx:
            other_depot_dists.append(np.linalg.norm(truck_positions[j] - depot_position))

    # Precompute depot distances for customers
    cust_depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)

    best_index = None
    best_savings = -1e9
    best_active_cost = 1e9
    fallback_index = None
    fallback_active_cost = 1e9

    if n_trucks == 1:
        # Single truck: always assign the one with smallest active_cost
        for i in range(len(available_customers)):
            active_cost = np.linalg.norm(current_position - available_customers[i]) + cust_depot_dists[i]
            if active_cost < best_active_cost:
                best_active_cost = active_cost
                best_index = i
        return best_index

    # Determine threshold based on active truck's depot distance relative to others
    max_other_depot = max(other_depot_dists)
    min_other_depot = min(other_depot_dists)
    if depot_dist_active >= max_other_depot:
        fallback_threshold = 1.3  # 30% slack, aggressive assignment
    elif depot_dist_active <= min_other_depot:
        fallback_threshold = 1.0  # no slack, strict waiting
    else:
        fallback_threshold = 1.1  # 10% slack

    for i in range(len(available_customers)):
        cust = available_customers[i]
        active_cost = np.linalg.norm(current_position - cust) + cust_depot_dists[i]

        # Compute min cost among other trucks
        min_other_cost = 1e9
        for j in range(n_trucks):
            if j == active_idx:
                continue
            other_cost = np.linalg.norm(truck_positions[j] - cust) + cust_depot_dists[i]
            if other_cost < min_other_cost:
                min_other_cost = other_cost

        if active_cost <= min_other_cost:
            savings = min_other_cost - active_cost
            if savings > best_savings or (savings == best_savings and active_cost < best_active_cost):
                best_savings = savings
                best_index = i
                best_active_cost = active_cost
        else:
            if active_cost <= fallback_threshold * min_other_cost:
                if active_cost < fallback_active_cost:
                    fallback_index = i
                    fallback_active_cost = active_cost

    if best_index is not None:
        return best_index
    elif fallback_index is not None:
        return fallback_index
    else:
        return None