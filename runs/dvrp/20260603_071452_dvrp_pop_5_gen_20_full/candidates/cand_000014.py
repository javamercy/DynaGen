import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    # distances from active truck to each customer
    d_active = np.linalg.norm(available_customers - current_position, axis=1)
    # find active truck index by matching position (assume unique nearest)
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    other_indices = [i for i in range(truck_positions.shape[0]) if i != active_idx]
    if len(other_indices) == 0:
        best = np.argmin(d_active)
        return int(best)
    other_trucks = truck_positions[other_indices]
    # min distance from any other truck to each customer
    d_other_min = np.min(np.linalg.norm(available_customers[:, None] - other_trucks[None], axis=2), axis=1)
    # adaptive threshold based on active truck's distance to depot
    depot_dist_active = np.linalg.norm(current_position - depot_position)
    max_depot_dist = np.max(np.linalg.norm(truck_positions - depot_position, axis=1))
    if max_depot_dist > 1e-9:
        threshold = 1.0 - 0.5 * (depot_dist_active / max_depot_dist)
    else:
        threshold = 1.0
    # ratio other_min / active, avoid division by zero
    ratio = np.where(d_active > 1e-9, d_other_min / d_active, np.inf)
    # select customers where active is sufficiently closer than others
    mask = ratio > threshold
    if np.any(mask):
        # among those, pick max ratio (prefer highest territorial advantage)
        candidates = np.where(mask, ratio, -np.inf)
        best = np.argmax(candidates)
    else:
        # fallback: nearest neighbor
        best = np.argmin(d_active)
    return int(best)