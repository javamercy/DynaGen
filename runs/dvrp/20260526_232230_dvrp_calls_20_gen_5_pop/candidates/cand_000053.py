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

    current_pos = current_position
    depot = depot_position
    other_trucks = truck_positions
    # Current return times
    current_own_return = dist(current_pos, depot)
    other_returns = dist(other_trucks, depot)
    current_max_return = max(current_own_return, np.max(other_returns) if len(other_returns) > 0 else 0.0)

    best_idx = None
    best_new_max = float('inf')
    best_new_return = float('inf')

    for i, customer in enumerate(available_customers):
        d_truck_cust = dist(current_pos, customer)
        d_cust_depot = dist(customer, depot)
        new_return = d_truck_cust + d_cust_depot
        # Max return if this truck serves customer
        new_max = max(new_return, np.max(other_returns) if len(other_returns) > 0 else 0.0)
        if new_max < best_new_max or (new_max == best_new_max and new_return < best_new_return):
            best_new_max = new_max
            best_new_return = new_return
            best_idx = i

    # Waiting condition: if best new max is more than 10% above current max, wait
    if best_new_max > 1.1 * current_max_return:
        return None
    else:
        return best_idx