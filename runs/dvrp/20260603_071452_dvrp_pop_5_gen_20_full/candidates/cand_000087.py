import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None

    n_trucks = truck_positions.shape[0]
    # Find active truck index
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")

    # Precompute distances from each customer to depot
    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)

    # Active truck's cost for each customer
    active_dists = np.linalg.norm(available_customers - current_position, axis=1)
    active_costs = active_dists + depot_dists

    # Single truck case
    if n_trucks == 1:
        return int(np.argmin(active_costs))

    # Compute costs for other trucks
    mask = np.ones(n_trucks, dtype=bool)
    mask[active_idx] = False
    other_positions = truck_positions[mask]
    other_dists = np.linalg.norm(
        available_customers[:, np.newaxis, :] - other_positions[np.newaxis, :, :], axis=2
    )
    other_costs = other_dists + depot_dists[:, np.newaxis]
    min_other = np.min(other_costs, axis=1)

    # Compute max completion and savings
    max_val = np.maximum(active_costs, min_other)
    savings = min_other - active_costs

    # Primary: active cost <= min other
    primary_mask = active_costs <= min_other
    if np.any(primary_mask):
        primary_max = max_val[primary_mask]
        primary_active = active_costs[primary_mask]
        # Lexsort: min max_val, then min active_cost
        order = np.lexsort((primary_active, primary_max))
        best_idx = np.where(primary_mask)[0][order[0]]
        return int(best_idx)

    # Fallback: active cost within 5% of min other
    fallback_mask = active_costs <= 1.05 * min_other
    if np.any(fallback_mask):
        fallback_max = max_val[fallback_mask]
        fallback_active = active_costs[fallback_mask]
        order = np.lexsort((fallback_active, fallback_max))
        best_idx = np.where(fallback_mask)[0][order[0]]
        return int(best_idx)

    # Otherwise wait
    return None