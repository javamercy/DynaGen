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
    # find active truck index
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")

    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)
    truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)

    active_depot_dist = truck_depot_dists[active_idx]
    median_depot = np.median(truck_depot_dists)

    # threshold for accepting when active is worse
    if active_depot_dist > 0:
        threshold = 1.0 + 0.2 * (median_depot / active_depot_dist)
        threshold = max(1.0, min(threshold, 2.0))
    else:
        threshold = 1.0

    best_index = None
    best_active_cost = np.inf
    best_depot_dist = np.inf

    for i in range(len(available_customers)):
        cust = available_customers[i]
        active_cost = np.linalg.norm(current_position - cust) + depot_dists[i]

        if n_trucks == 1:
            if active_cost < best_active_cost:
                best_index = i
                best_active_cost = active_cost
            continue

        # compute min cost among other trucks
        min_other = np.inf
        for j in range(n_trucks):
            if j == active_idx:
                continue
            cost = np.linalg.norm(truck_positions[j] - cust) + depot_dists[i]
            if cost < min_other:
                min_other = cost

        # decide if active should take this customer
        if active_cost <= min_other:
            # active is best or equal
            if active_cost < best_active_cost or (active_cost == best_active_cost and depot_dists[i] < best_depot_dist):
                best_index = i
                best_active_cost = active_cost
                best_depot_dist = depot_dists[i]
        elif active_cost <= threshold * min_other:
            # active is worse but within threshold
            if active_cost < best_active_cost or (active_cost == best_active_cost and depot_dists[i] < best_depot_dist):
                best_index = i
                best_active_cost = active_cost
                best_depot_dist = depot_dists[i]

    return best_index