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

    # Current truck's distance to depot
    current_dist_to_depot = np.linalg.norm(current_position - depot_position)

    # Other trucks
    mask = ~np.all(truck_positions == current_position, axis=1)
    other_positions = truck_positions[mask]

    if other_positions.shape[0] == 0:
        # No other trucks: simply minimize own return time
        best_idx = None
        best_return = float('inf')
        for i, customer in enumerate(available_customers):
            d_truck_cust = np.linalg.norm(current_position - customer)
            d_cust_depot = np.linalg.norm(customer - depot_position)
            new_return = d_truck_cust + d_cust_depot
            if new_return < best_return:
                best_return = new_return
                best_idx = i
        return best_idx

    # Compute max return of other trucks (distance to depot)
    other_returns = np.linalg.norm(other_positions - depot_position, axis=1)
    max_other_return = np.max(other_returns)

    # Current max return if truck does nothing (returns directly)
    current_max = max(current_dist_to_depot, max_other_return)

    best_idx = None
    best_new_max = float('inf')
    best_new_return = float('inf')

    for i, customer in enumerate(available_customers):
        d_truck_cust = np.linalg.norm(current_position - customer)
        d_cust_depot = np.linalg.norm(customer - depot_position)
        new_return = d_truck_cust + d_cust_depot
        new_max = max(new_return, max_other_return)
        # Only consider if new_max does not exceed current_max
        if new_max <= current_max:
            # Among these, prefer smaller new_max (and smaller new_return as tie-breaker)
            if (new_max < best_new_max) or (new_max == best_new_max and new_return < best_new_return):
                best_new_max = new_max
                best_new_return = new_return
                best_idx = i

    if best_idx is not None:
        return best_idx
    else:
        return None