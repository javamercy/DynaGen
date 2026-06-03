import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    
    n_trucks = truck_positions.shape[0]
    # Find active truck index
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")
    
    # Precompute depot distances for customers
    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)
    
    # Active truck distances and round trips
    active_dists = np.linalg.norm(available_customers - current_position, axis=1)
    active_rounds = active_dists + depot_dists
    
    if n_trucks == 1:
        best_idx = np.argmin(active_rounds)
        return int(best_idx)
    
    # Other trucks positions
    mask = np.ones(n_trucks, dtype=bool)
    mask[active_idx] = False
    other_positions = truck_positions[mask]
    # Other trucks distances and round trips for each customer
    other_dists = np.linalg.norm(
        available_customers[:, np.newaxis, :] - other_positions[np.newaxis, :, :], axis=2
    )
    other_rounds = other_dists + depot_dists[:, np.newaxis]
    best_other_rounds = np.min(other_rounds, axis=1)
    
    # Eligible: active round trip <= best other round trip
    eligible = active_rounds <= best_other_rounds
    if np.any(eligible):
        eligible_indices = np.where(eligible)[0]
        # Among eligible, pick the one with smallest active round trip
        best_idx_in_eligible = np.argmin(active_rounds[eligible])
        best_idx = eligible_indices[best_idx_in_eligible]
        return int(best_idx)
    else:
        return None