import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None

    # Identify active truck index
    active_idx = None
    for i, pos in enumerate(truck_positions):
        if np.allclose(pos, current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")

    # Current distances to depot for all trucks
    truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max = np.max(truck_depot_dists)

    best_idx = None
    best_candidate_max = np.inf
    best_active_completion = np.inf

    for i, cust in enumerate(available_customers):
        active_completion = (np.linalg.norm(current_position - cust) +
                             np.linalg.norm(cust - depot_position))
        # Other trucks' distances remain unchanged (optimistic)
        other_distances = [truck_depot_dists[j] for j in range(len(truck_positions)) if j != active_idx]
        candidate_max = max(active_completion, max(other_distances))

        # Wait condition: reject if candidate_max exceeds current_max by more than 10%
        if candidate_max > current_max * 1.1:
            continue

        # Pick the one with smallest candidate_max; break ties by active_completion
        if candidate_max < best_candidate_max or (
            candidate_max == best_candidate_max and active_completion < best_active_completion
        ):
            best_candidate_max = candidate_max
            best_active_completion = active_completion
            best_idx = i

    return best_idx