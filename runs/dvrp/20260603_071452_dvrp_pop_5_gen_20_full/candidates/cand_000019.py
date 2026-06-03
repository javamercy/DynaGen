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

    threshold = 1.2  # allow up to 20% worse than best
    best_idx = None
    best_ratio = np.inf
    best_active_cost = np.inf

    for i in range(len(available_customers)):
        active_cost = np.linalg.norm(current_position - available_customers[i]) + depot_dist[i]

        if n_trucks == 1:
            # only one truck, always serve the nearest
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

        # avoid division by zero
        if min_other == 0:
            ratio = 1.0 if active_cost == 0 else np.inf
        else:
            ratio = active_cost / min_other

        if ratio <= threshold:
            # eligible, select based on ratio then active cost
            if ratio < best_ratio or (ratio == best_ratio and active_cost < best_active_cost):
                best_ratio = ratio
                best_active_cost = active_cost
                best_idx = i

    return best_idx