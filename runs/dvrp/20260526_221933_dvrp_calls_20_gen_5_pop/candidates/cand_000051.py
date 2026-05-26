import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if len(available_customers) == 0:
        return None
    dist_current = np.linalg.norm(available_customers - current_position, axis=1)
    dist_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    mask = ~np.all(truck_positions == current_position, axis=1)
    other_trucks = truck_positions[mask]
    if len(other_trucks) == 0:
        nearest_other = np.full(len(available_customers), np.inf)
        dyn_weight = 0.3
    else:
        diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
        dist_other = np.linalg.norm(diff, axis=2)
        nearest_other = np.min(dist_other, axis=1)
        other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
        mean_other_depot = other_depot_dists.mean()
        if mean_other_depot > 0:
            avg_avail_depot = dist_depot.mean()
            dyn_weight = 0.3 * avg_avail_depot / mean_other_depot
            dyn_weight = np.clip(dyn_weight, 0.1, 1.0)
        else:
            dyn_weight = 0.3
    score = dist_current - nearest_other + dyn_weight * dist_depot
    best_idx = np.argmin(score)
    return int(best_idx)