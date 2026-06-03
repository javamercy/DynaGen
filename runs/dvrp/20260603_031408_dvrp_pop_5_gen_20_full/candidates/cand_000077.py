import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None

    # distances from all trucks to depot
    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # identify active truck index (by position match; assume unique)
    active_idx = np.where(np.all(np.isclose(truck_positions, current_position), axis=1))[0][0]
    active_dist = truck_to_depot[active_idx]
    # other trucks' distances
    other_mask = np.ones(truck_positions.shape[0], dtype=bool)
    other_mask[active_idx] = False
    other_dist = truck_to_depot[other_mask]
    max_other = np.max(other_dist) if other_dist.size > 0 else 0.0
    min_other = np.min(other_dist) if other_dist.size > 0 else 0.0

    # distances to each available customer from active truck and to depot
    dist_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    new_return = dist_to_cust + cust_to_depot

    # candidate max as in cand_000015: max(new_return, max_other)
    candidate_max = np.maximum(new_return, max_other)
    best_idx = int(np.argmin(candidate_max))
    # tie-break by smaller new_return
    ties = np.where(candidate_max == candidate_max[best_idx])[0]
    if len(ties) > 1:
        best_idx = int(ties[np.argmin(new_return[ties])])
    best_new_return = new_return[best_idx]

    # waiting condition: if active truck is closest to depot and best customer is far relative to active_dist
    if active_dist < min_other and best_new_return > active_dist + 0.5 * max_other:
        return None
    else:
        return best_idx