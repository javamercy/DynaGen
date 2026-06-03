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

    # compute depot distances of all trucks
    truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    median_depot = np.median(truck_depot_dists)
    active_depot_dist = truck_depot_dists[active_idx]

    # adaptive base threshold: increases with number of available customers
    n_avail = len(available_customers)
    base_threshold = 1.1 + 0.1 * min(n_avail / 10, 5)  # range 1.1 to 1.6

    best_index = None
    best_savings = -np.inf
    best_active_cost = np.inf
    fallback_index = None
    fallback_active_cost = np.inf
    # track best fallback to break ties by depot distance
    best_fallback_depot_dist = np.inf

    for i in range(n_avail):
        cust = available_customers[i]
        active_cost = np.linalg.norm(current_position - cust) + depot_dists[i]

        if n_trucks == 1:
            if active_cost < best_active_cost or (active_cost == best_active_cost and depot_dists[i] < best_fallback_depot_dist):
                best_index = i
                best_active_cost = active_cost
                best_fallback_depot_dist = depot_dists[i]
            continue

        # compute min cost among other trucks and find best other truck
        min_other = np.inf
        best_other_idx = -1
        for j in range(n_trucks):
            if j == active_idx:
                continue
            cost = np.linalg.norm(truck_positions[j] - cust) + depot_dists[i]
            if cost < min_other:
                min_other = cost
                best_other_idx = j

        savings = min_other - active_cost

        if savings >= 0:
            # active is at least as good as any other truck
            if (savings > best_savings or 
                (savings == best_savings and active_cost < best_active_cost)):
                best_savings = savings
                best_index = i
                best_active_cost = active_cost
        else:
            # active is worse; fallback with modulated threshold
            best_other_depot = truck_depot_dists[best_other_idx]
            if median_depot > 0 and active_depot_dist > 0:
                # alpha larger when active truck is close to depot
                alpha = 0.1 * (median_depot / (active_depot_dist + 1e-6))
                factor = (best_other_depot - median_depot) / median_depot
                modulated_threshold = base_threshold * (1 + alpha * factor)
            else:
                modulated_threshold = base_threshold
            modulated_threshold = max(1.0, min(modulated_threshold, 2.0))
            if active_cost <= modulated_threshold * min_other:
                # prefer lower active cost, tie-break by depot distance
                if (active_cost < fallback_active_cost or 
                    (active_cost == fallback_active_cost and depot_dists[i] < best_fallback_depot_dist)):
                    fallback_index = i
                    fallback_active_cost = active_cost
                    best_fallback_depot_dist = depot_dists[i]

    if best_index is not None:
        return best_index
    elif fallback_index is not None:
        return fallback_index
    else:
        return None