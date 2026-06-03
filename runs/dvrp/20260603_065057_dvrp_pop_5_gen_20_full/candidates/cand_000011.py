import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None

    # find index of current truck in truck_positions (assuming exact match)
    truck_idx = None
    for i, pos in enumerate(truck_positions):
        if np.array_equal(pos, current_position):
            truck_idx = i
            break
    if truck_idx is None:
        # fallback: use closest truck
        dists = np.linalg.norm(truck_positions - current_position, axis=1)
        truck_idx = int(np.argmin(dists))

    # precompute distances from all trucks to all customers
    n_trucks = len(truck_positions)
    n_cust = len(available_customers)
    dists = np.zeros((n_trucks, n_cust))
    for t in range(n_trucks):
        dists[t] = np.linalg.norm(available_customers - truck_positions[t], axis=1)

    # find nearest truck for each customer
    nearest_truck = np.argmin(dists, axis=0)  # shape (n_cust,)
    nearest_dist = np.min(dists, axis=0)      # shape (n_cust,)

    # only consider customers whose nearest truck is the current truck
    owned_mask = (nearest_truck == truck_idx)
    if not np.any(owned_mask):
        return None

    # among owned, pick the one with largest nearest distance (most isolated)
    owned_indices = np.where(owned_mask)[0]
    best_idx = owned_indices[np.argmax(nearest_dist[owned_indices])]

    return int(best_idx)