def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if len(available_customers) == 0:
        return None
    n_trucks = len(truck_positions)
    dist_current = np.linalg.norm(available_customers - current_position, axis=1)
    dist_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    truck_return = current_time + dist_current + dist_depot
    mask = np.all(truck_positions == current_position, axis=1)
    if np.any(mask):
        current_idx = np.where(mask)[0][0]
    else:
        current_idx = None
    if n_trucks == 1:
        best_idx = np.argmin(truck_return)
        return int(best_idx)
    other_mask = np.ones(n_trucks, dtype=bool)
    other_mask[current_idx] = False
    other_returns = current_time + np.linalg.norm(truck_positions[other_mask] - depot_position, axis=1)
    max_other = np.max(other_returns)
    max_return = np.maximum(truck_return, max_other)
    min_max = np.min(max_return)
    candidates = np.where(max_return == min_max)[0]
    if len(candidates) == 1:
        best_idx = candidates[0]
    else:
        other_trucks = truck_positions[other_mask]
        diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
        dist_other = np.linalg.norm(diff, axis=2)
        nearest_other = np.min(dist_other, axis=1)
        parent_score = dist_current - nearest_other + 0.3 * dist_depot
        candidate_scores = parent_score[candidates]
        best_in_candidates = np.argmin(candidate_scores)
        best_idx = candidates[best_in_candidates]
    return int(best_idx)