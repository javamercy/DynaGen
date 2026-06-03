import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    # Euclidean distances from active truck to each customer
    d_active = np.linalg.norm(available_customers - current_position, axis=1)
    # Distance from each other truck to each customer (broadcast)
    # Find index of active truck (assume first match since positions may be duplicate? use closest)
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    other_indices = [i for i in range(truck_positions.shape[0]) if i != active_idx]
    other_trucks = truck_positions[other_indices]
    if len(other_indices) == 0:
        # Only one truck, just pick nearest
        best = np.argmin(d_active)
        return int(best)
    # Compute min distance from any other truck to each customer
    d_other_min = np.min(np.linalg.norm(available_customers[:, None] - other_trucks[None], axis=2), axis=1)
    # Ratio: other_min / active (avoid division by zero)
    ratio = np.where(d_active > 1e-9, d_other_min / d_active, np.inf)
    # Where ratio > 1 (active is closer than any other), pick max ratio; else pick min d_active
    mask = ratio > 1.0
    if np.any(mask):
        # Among those where active is closer, pick max ratio (prefer customers where others are relatively far)
        candidates = np.where(mask, ratio, -np.inf)
        best = np.argmax(candidates)
    else:
        # Fallback: pick nearest to active
        best = np.argmin(d_active)
    return int(best)