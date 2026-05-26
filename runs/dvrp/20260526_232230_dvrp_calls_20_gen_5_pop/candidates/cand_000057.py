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

    # distances from each truck to depot
    other_returns = dist(truck_positions, depot_position)
    # current max return time if this truck waits (stays at current position)
    current_return_if_wait = dist(current_position, depot_position)
    current_max = max(np.max(other_returns), current_return_if_wait)

    best_idx = None
    best_new_max = float('inf')
    best_new_return = float('inf')

    for i, customer in enumerate(available_customers):
        d_to_cust = dist(current_position, customer)
        d_cust_to_depot = dist(customer, depot_position)
        new_return = d_to_cust + d_cust_to_depot
        new_max = max(new_return, np.max(other_returns))

        # If new_max > current_max for every customer, we will wait (return None)
        # We'll keep track of the best if there is any that does not increase max
        if new_max <= current_max:
            # candidate does not worsen max
            if (new_max < best_new_max or
                (new_max == best_new_max and new_return < best_new_return)):
                best_new_max = new_max
                best_new_return = new_return
                best_idx = i

    # If no customer keeps max <= current_max, return None to wait
    return best_idx