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
    # identify active truck index
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")
    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)
    active_dists = np.linalg.norm(available_customers - current_position, axis=1)
    active_costs = active_dists + depot_dists
    if n_trucks == 1:
        best_idx = np.argmin(active_costs)
        return int(best_idx)
    mask = np.ones(n_trucks, dtype=bool)
    mask[active_idx] = False
    other_positions = truck_positions[mask]
    other_dists = np.linalg.norm(
        available_customers[:, np.newaxis, :] - other_positions[np.newaxis, :, :], axis=2
    )
    other_costs = other_dists + depot_dists[:, np.newaxis]
    min_other = np.min(other_costs, axis=1)
    savings = min_other - active_costs
    # primary: active <= best other
    primary_mask = active_costs <= min_other
    if np.any(primary_mask):
        primary_savings = savings[primary_mask]
        primary_active = active_costs[primary_mask]
        order = np.lexsort((primary_active, -primary_savings))
        best_primary_idx = np.where(primary_mask)[0][order[0]]
        return int(best_primary_idx)
    # fallback: adapt threshold based on number of available customers
    n_avail = len(available_customers)
    if n_avail > 10:
        threshold = 1.05
    else:
        threshold = 1.2
    fallback_mask = active_costs <= threshold * min_other
    if np.any(fallback_mask):
        fallback_active = active_costs[fallback_mask]
        best_fallback_idx = np.where(fallback_mask)[0][np.argmin(fallback_active)]
        return int(best_fallback_idx)
    return None