import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    # identify own truck index
    idx_self = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    n_trucks = truck_positions.shape[0]
    # precompute distances
    self_dists = np.linalg.norm(available_customers - current_position, axis=1)
    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)
    best_score = -np.inf
    best_idx = -1
    for i, cust in enumerate(available_customers):
        # min distance to other trucks
        dists_to_trucks = np.linalg.norm(truck_positions - cust, axis=1)
        dists_to_trucks[idx_self] = np.inf
        min_other = np.min(dists_to_trucks)
        score = min_other - 0.5 * self_dists[i] - 0.3 * depot_dists[i]
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx