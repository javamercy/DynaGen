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

    active_return = dist(current_position, depot_position)
    all_returns = dist(truck_positions, depot_position)
    # find index of active truck
    active_mask = np.all(np.isclose(truck_positions, current_position), axis=1)
    if np.sum(active_mask) == 0:
        # fallback: treat all as others (should not happen)
        max_others = np.max(all_returns)
    else:
        other_returns = all_returns[~active_mask]
        max_others = np.max(other_returns) if other_returns.size > 0 else 0.0

    current_max = max(active_return, max_others)

    # waiting condition: only if there are other trucks
    if truck_positions.shape[0] > 1:
        all_increase = True
        for customer in available_customers:
            d_to_cust = dist(current_position, customer)
            d_cust_to_depot = dist(customer, depot_position)
            new_return_active = d_to_cust + d_cust_to_depot
            if new_return_active <= current_max:
                all_increase = False
                break
        if all_increase:
            return None

    best_idx = None
    best_new_max = float('inf')
    best_new_return = float('inf')

    for i, customer in enumerate(available_customers):
        d_to_cust = dist(current_position, customer)
        d_cust_depot = dist(customer, depot_position)
        new_return = d_to_cust + d_cust_depot
        new_max = max(new_return, max_others)

        if (new_max < best_new_max) or (new_max == best_new_max and new_return < best_new_return):
            best_new_max = new_max
            best_new_return = new_return
            best_idx = i

    return best_idx