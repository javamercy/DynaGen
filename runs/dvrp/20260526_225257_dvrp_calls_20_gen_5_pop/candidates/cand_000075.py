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
    # compute own_score (current->customer + customer->depot)
    current_to_customer = np.linalg.norm(available_customers - current_position, axis=1)
    customer_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    own_score = current_to_customer + customer_to_depot
    best_own_idx = np.argmin(own_score)
    best_own_val = own_score[best_own_idx]
    # identify other trucks
    mask = np.all(np.abs(truck_positions - current_position) < 1e-8, axis=1)
    other_trucks = truck_positions[~mask]
    if len(other_trucks) > 0:
        # compute other trucks' distances to depot
        other_depot_dist = np.linalg.norm(other_trucks - depot_position, axis=1)
        avg_other_depot = np.mean(other_depot_dist)
        # wait if best own_score is small relative to average other truck depot distance
        if best_own_val < 0.3 * avg_other_depot:
            return None
        # otherwise compute full score with isolation bonus (beta=0.3)
        dist_to_other = np.linalg.norm(available_customers[:, None, :] - other_trucks[None, :, :], axis=2)
        nearest_other = np.min(dist_to_other, axis=1)
        beta = 0.3
        score = own_score - beta * nearest_other
        best_idx = np.argmin(score)
        return int(best_idx)
    else:
        # no other trucks, just use own_score
        return int(best_own_idx)