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
    # Identify active truck index
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")

    active_dists = np.linalg.norm(available_customers - current_position, axis=1)
    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)
    active_costs = active_dists + depot_dists

    if n_trucks == 1:
        return int(np.argmin(active_costs))

    # Costs for other trucks
    mask = np.ones(n_trucks, dtype=bool)
    mask[active_idx] = False
    other_positions = truck_positions[mask]
    other_dists = np.linalg.norm(
        available_customers[:, np.newaxis, :] - other_positions[np.newaxis, :, :], axis=2
    )
    other_costs = other_dists + depot_dists[:, np.newaxis]
    min_other = np.min(other_costs, axis=1)

    # Idle trucks at depot
    truck_depot_dist = np.linalg.norm(truck_positions - depot_position, axis=1)
    idle_mask = truck_depot_dist < 1e-6
    if np.any(idle_mask):
        idle_other_mask = idle_mask.copy()
        idle_other_mask[active_idx] = False
        if np.any(idle_other_mask):
            idle_positions = truck_positions[idle_other_mask]
            idle_dists = np.linalg.norm(
                available_customers[:, np.newaxis, :] - idle_positions[np.newaxis, :, :], axis=2
            )
            idle_costs = idle_dists + depot_dists[:, np.newaxis]
            min_other_idle = np.min(idle_costs, axis=1)
        else:
            min_other_idle = np.full(len(available_customers), np.inf)
    else:
        min_other_idle = np.full(len(available_customers), np.inf)

    # Savings and imbalance penalty
    base_savings = min_other - active_costs
    IMBALANCE_PENALTY = 0.5
    penalty = np.where(
        (min_other_idle < active_costs) & np.isfinite(min_other_idle),
        IMBALANCE_PENALTY * (active_costs - min_other_idle),
        0.0
    )
    adjusted_savings = base_savings - penalty

    # Tier: primary (2), fallback (1), or invalid (-1)
    fallback_threshold = -0.1 * min_other
    tier = np.where(
        base_savings >= 0,
        2,
        np.where(base_savings >= fallback_threshold, 1, -1)
    )

    valid = tier >= 1
    if not np.any(valid):
        return None

    valid_indices = np.where(valid)[0]
    sub_tier = tier[valid]
    sub_adj = adjusted_savings[valid]
    sub_active = active_costs[valid]

    # Lexsort: descending tier, descending adjusted_savings, ascending active_cost
    order = np.lexsort((sub_active, -sub_adj, -sub_tier))
    best_index = valid_indices[order[0]]
    return int(best_index)