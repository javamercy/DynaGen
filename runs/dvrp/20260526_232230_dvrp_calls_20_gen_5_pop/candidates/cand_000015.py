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
        return np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1)

    current_pos = current_position
    depot = depot_position
    other_trucks = truck_positions

    # Current return times
    current_truck_return = dist(current_pos, depot)
    other_returns = dist(other_trucks, depot)
    max_other = np.max(other_returns)
    current_max = max(current_truck_return, max_other)

    best_idx = None
    best_new_max = float('inf')
    best_new_return = float('inf')

    for i, customer in enumerate(available_customers):
        d_truck_cust = dist(current_pos, customer)
        d_cust_depot = dist(customer, depot)
        new_return = d_truck_cust + d_cust_depot
        new_max = max(new_return, max_other)
        if new_max < best_new_max - 1e-12 or (abs(new_max - best_new_max) < 1e-12 and new_return < best_new_return):
            best_new_max = new_max
            best_new_return = new_return
            best_idx = i

    if best_new_max > current_max + 1e-12:
        return None
    return best_idx