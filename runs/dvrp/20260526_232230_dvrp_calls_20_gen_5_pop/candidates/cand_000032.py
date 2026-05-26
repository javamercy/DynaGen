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
    current_truck_return = dist(current_pos, depot)
    other_returns = dist(other_trucks, depot)
    current_max_return = max(current_truck_return, np.max(other_returns) if len(other_returns) > 0 else -np.inf)

    best_idx = None
    best_new_max = float('inf')
    best_new_return = float('inf')

    for i, customer in enumerate(available_customers):
        d_truck_cust = dist(current_pos, customer)
        d_cust_depot = dist(customer, depot)
        new_return = d_truck_cust + d_cust_depot
        new_max = max(new_return, np.max(other_returns) if len(other_returns) > 0 else -np.inf)
        if new_max < best_new_max or (new_max == best_new_max and new_return < best_new_return):
            best_new_max = new_max
            best_new_return = new_return
            best_idx = i

    # Wait if this truck is the bottleneck (its current return equals current max)
    # and serving any customer increases max (which it does because new_return >= current_truck_return)
    # and there are other trucks that might serve customers
    if current_truck_return >= current_max_return - 1e-9:
        # check if best_new_max > current_max_return (it will be unless customer on exact path, improbable)
        if best_new_max > current_max_return + 1e-9:
            return None

    return best_idx