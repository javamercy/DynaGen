import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None

    n_trucks = len(truck_positions)
    # Find active truck index
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")

    # Precompute round-trip costs: distance truck->customer + customer->depot
    # shape (n_trucks, n_available)
    truck_to_cust = np.linalg.norm(truck_positions[:, np.newaxis, :] - available_customers[np.newaxis, :, :], axis=2)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)  # (n_available,)
    costs = truck_to_cust + cust_to_depot[np.newaxis, :]  # (n_trucks, n_available)

    # For each customer, find best truck (minimum cost)
    best_truck = np.argmin(costs, axis=0)  # indices of best truck for each customer

    # Get customer indices assigned to the active truck
    assigned_mask = best_truck == active_idx
    if not np.any(assigned_mask):
        return None

    assigned_indices = np.where(assigned_mask)[0]
    assigned_customers = available_customers[assigned_mask]

    # Among assigned customers, pick the one nearest to current position
    distances = np.linalg.norm(assigned_customers - current_position, axis=1)
    best_rel_idx = int(np.argmin(distances))
    return int(assigned_indices[best_rel_idx])