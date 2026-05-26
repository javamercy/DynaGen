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
    mask = np.all(truck_positions == current_position, axis=1)
    if mask.any():
        other_trucks = truck_positions[~mask]
    else:
        other_trucks = truck_positions
    # time-dependent depot coefficient: starts at 0.4, increases to 0.6 asymptotically
    depot_coeff = 0.4 + 0.2 * (current_time / (current_time + 10.0))
    if len(other_trucks) == 0:
        score = dist_current + depot_coeff * dist_depot
    else:
        diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
        dist_other = np.linalg.norm(diff, axis=2)
        nearest_other = np.min(dist_other, axis=1)
        score = dist_current - nearest_other + depot_coeff * dist_depot
    best_idx = np.argmin(score)
    return int(best_idx)