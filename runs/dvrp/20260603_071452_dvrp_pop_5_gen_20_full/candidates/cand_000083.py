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

    # compute median depot distance of all trucks
    truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    median_depot = np.median(truck_depot_dists)
    # active truck's distance to depot
    active_truck_depot = np.linalg.norm(current_position - depot_position)

    best_index = None
    best_savings = -np.inf
    best_active_cost = np.inf
    fallback_index = None
    fallback_active_cost = np.inf

    n_avail = len(available_customers)
    if n_avail <= 5:
        base_threshold = 1.2
    else:
        base_threshold = 1.1

    for i in range(n_avail):
        cust = available_customers[i]
        active_cost = np.linalg.norm(current_position - cust) + depot_dists[i]

        # single truck case
        if n_trucks == 1:
            if active_cost < best_active_cost:
                best_index = i
                best_active_cost = active_cost
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
            if savings > best_savings or (savings == best_savings and active_cost < best_active_cost):
                best_savings = savings
                best_index = i
                best_active_cost = active_cost
        else:
            # active is worse than best other truck; consider fallback with modulated threshold
            # compute modulation factor based on active truck's depot distance relative to median
            if median_depot > 0:
                factor = (active_truck_depot - median_depot) / median_depot
                alpha = 0.2
                modulated_threshold = base_threshold * (1 + alpha * factor)
            else:
                modulated_threshold = base_threshold
            # cap threshold to avoid extremely poor assignments
            modulated_threshold = max(1.0, min(modulated_threshold, 2.0))
            if active_cost <= modulated_threshold * min_other:
                if active_cost < fallback_active_cost:
                    fallback_index = i
                    fallback_active_cost = active_cost

    if best_index is not None:
        return best_index
    elif fallback_index is not None:
        return fallback_index
    else:
        return None