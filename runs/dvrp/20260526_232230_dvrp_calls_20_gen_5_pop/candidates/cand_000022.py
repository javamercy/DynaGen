import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b, axis=-1)

    # Distance from current truck to depot
    current_return = dist(current_position, depot_position)

    # Compute other trucks' returns excluding current truck
    # truck_positions includes all trucks; find which ones are not current_position
    # Use tolerance to check equality
    tol = 1e-9
    mask = np.any(np.abs(truck_positions - current_position) > tol, axis=1)
    other_returns = dist(truck_positions[mask], depot_position) if np.any(mask) else np.array([])
    other_max = np.max(other_returns) if other_returns.size > 0 else 0.0
    current_max = max(current_return, other_max)

    best_idx = None
    best_new_max = float('inf')
    best_new_return = float('inf')

    for i, customer in enumerate(available_customers):
        d_truck_cust = dist(current_position, customer)
        d_cust_depot = dist(customer, depot_position)
        new_return = d_truck_cust + d_cust_depot
        new_max = max(new_return, other_max)
        # Minimize new_max; tie-break by new_return
        if new_max < best_new_max or (new_max == best_new_max and new_return < best_new_return):
            best_new_max = new_max
            best_new_return = new_return
            best_idx = i

    # Wait if serving increases max return by more than 15%
    if best_new_max > 1.15 * current_max:
        return None
    return best_idx