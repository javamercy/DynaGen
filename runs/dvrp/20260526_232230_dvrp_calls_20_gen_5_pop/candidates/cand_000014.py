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
    def dist(a, b):
        return np.linalg.norm(a - b, axis=-1)
    # Compute all return times to depot
    all_returns = dist(truck_positions, depot_position)
    # Find index of current truck by exact position match
    mask = np.all(truck_positions == current_position, axis=1)
    current_idx = np.where(mask)[0][0]
    current_truck_return = all_returns[current_idx]
    other_returns = np.delete(all_returns, current_idx)
    max_other = np.max(other_returns) if len(other_returns) > 0 else -np.inf
    current_max = max(current_truck_return, max_other)
    best_idx = None
    best_new_max = float('inf')
    best_new_return = float('inf')
    for i, customer in enumerate(available_customers):
        d_truck_cust = dist(current_position, customer)
        d_cust_depot = dist(customer, depot_position)
        new_return = d_truck_cust + d_cust_depot
        new_max = max(new_return, max_other)
        if new_max < best_new_max or (new_max == best_new_max and new_return < best_new_return):
            best_new_max = new_max
            best_new_return = new_return
            best_idx = i
    # Wait if best customer increases max return by more than 10%
    if best_new_max > current_max * 1.1:
        return None
    return best_idx