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

    depot = depot_position
    current_truck_return = dist(current_position, depot)
    other_returns = dist(truck_positions, depot)
    max_other = np.max(other_returns)
    current_max = max(current_truck_return, max_other)

    best_idx = None
    best_new_return = float('inf')

    for i, customer in enumerate(available_customers):
        d_truck_cust = dist(current_position, customer)
        d_cust_depot = dist(customer, depot)
        new_return = d_truck_cust + d_cust_depot
        new_max = max(new_return, max_other)
        if new_max <= current_max:  # does not increase the max
            if new_return < best_new_return:
                best_new_return = new_return
                best_idx = i

    if best_idx is None:
        return None
    else:
        return best_idx