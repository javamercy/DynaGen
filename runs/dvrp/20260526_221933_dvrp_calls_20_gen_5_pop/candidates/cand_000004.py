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

    n_trucks = len(truck_positions)
    active_idx = np.where((truck_positions == current_position).all(axis=1))[0][0]

    # If only one truck, serve nearest customer
    if n_trucks == 1:
        dists = np.linalg.norm(available_customers - current_position, axis=1)
        return int(np.argmin(dists))

    # Multiple trucks
    dist_active = np.linalg.norm(available_customers - current_position, axis=1)
    dist_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)

    # Compute min distance to any other truck for each customer
    other_positions = np.delete(truck_positions, active_idx, axis=0)
    # For each customer, compute distance to each other truck and take min
    diff = available_customers[:, np.newaxis, :] - other_positions[np.newaxis, :, :]  # shape (n_cust, n_other, 2)
    dist_other = np.linalg.norm(diff, axis=2)  # shape (n_cust, n_other)
    min_other = np.min(dist_other, axis=1)

    # Owned customers: distance to active <= min distance to other (with tolerance)
    owned_mask = dist_active <= min_other + 1e-9

    if np.any(owned_mask):
        # Among owned, pick the one farthest from depot
        owned_depot_dists = dist_to_depot[owned_mask]
        best_local_idx = np.argmax(owned_depot_dists)
        # Get overall index
        overall_idx = np.where(owned_mask)[0][best_local_idx]
        return int(overall_idx)
    else:
        # No owned customer, wait
        return None