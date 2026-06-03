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
    # identify active truck index
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")

    # precompute distances from each customer to depot
    delta = available_customers - depot_position
    depot_dist = np.linalg.norm(delta, axis=1)

    # compute distances from each truck to depot
    truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    active_depot_dist = truck_depot_dists[active_idx]
    avg_depot_dist = np.mean(truck_depot_dists)
    eps = 1e-6
    # adaptive threshold: closer to depot -> higher threshold (more patient), farther -> lower (more aggressive)
    threshold = 1.0 + (2.0 - 1.0) * (avg_depot_dist / (active_depot_dist + avg_depot_dist + eps))
    threshold = np.clip(threshold, 1.0, 2.0)

    best_idx = None
    best_ratio = np.inf
    best_active_cost = np.inf

    for i in range(len(available_customers)):
        active_cost = np.linalg.norm(current_position - available_customers[i]) + depot_dist[i]

        if n_trucks == 1:
            if active_cost < best_active_cost:
                best_idx = i
                best_active_cost = active_cost
            continue

        # compute costs for other trucks
        other_costs = []
        for j in range(n_trucks):
            if j == active_idx:
                continue
            cost = np.linalg.norm(truck_positions[j] - available_customers[i]) + depot_dist[i]
            other_costs.append(cost)
        min_other = min(other_costs)

        if min_other == 0:
            ratio = 1.0 if active_cost == 0 else np.inf
        else:
            ratio = active_cost / min_other

        if ratio <= threshold:
            if ratio < best_ratio or (ratio == best_ratio and active_cost < best_active_cost):
                best_ratio = ratio
                best_active_cost = active_cost
                best_idx = i

    return best_idx