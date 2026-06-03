import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    d_active = np.linalg.norm(available_customers - current_position, axis=1)
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    other_indices = [i for i in range(truck_positions.shape[0]) if i != active_idx]
    if len(other_indices) == 0:
        best = np.argmin(d_active)
        return int(best)
    other_trucks = truck_positions[other_indices]
    d_other_min = np.min(np.linalg.norm(available_customers[:, None] - other_trucks[None], axis=2), axis=1)
    depot_dist = np.linalg.norm(available_customers - depot_position, axis=1)
    ratio = np.where(d_active > 1e-9, d_other_min / d_active, np.inf)
    mask = ratio > 1.0
    if np.any(mask):
        scores = np.where(mask, ratio * depot_dist, -np.inf)
        best = np.argmax(scores)
    else:
        best = np.argmin(d_active)
    return int(best)