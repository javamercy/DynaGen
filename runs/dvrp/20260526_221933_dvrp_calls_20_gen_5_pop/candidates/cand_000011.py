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
    else:
        diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
        dist_other = np.linalg.norm(diff, axis=2)
        nearest_other = np.min(dist_other, axis=1)
    score = 0.7 * dist_current + 0.3 * dist_depot - 0.4 * nearest_other
    best_idx = np.argmin(score)
    # Dynamic threshold: wait if best score too high relative to truck-depot distance
    truck_to_depot = np.linalg.norm(current_position - depot_position)
    if score[best_idx] > 1.5 * truck_to_depot:
        return None
    return int(best_idx)