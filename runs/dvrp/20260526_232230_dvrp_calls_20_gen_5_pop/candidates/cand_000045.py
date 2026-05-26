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

    # Current return times for all trucks (including the active one) if they return directly.
    current_returns = dist(truck_positions, depot_position)
    current_max = np.max(current_returns)

    other_returns = np.delete(current_returns, np.where((truck_positions == current_position).all(axis=1))[0], axis=0) if truck_positions.shape[0] > 1 else np.array([])
    other_max = np.max(other_returns) if other_returns.size > 0 else -np.inf

    best_idx = None
    best_new_max = float('inf')
    best_new_return = float('inf')

    for i, customer in enumerate(available_customers):
        d_truck_cust = dist(current_position, customer)
        d_cust_depot = dist(customer, depot_position)
        new_return = d_truck_cust + d_cust_depot
        new_max = max(new_return, other_max)

        # Check if all customers increase max; if so, return None later.
        # We'll track the minimum new_max among those that do not increase?
        # Actually we want: if any customer does NOT increase max, we can choose.
        # So we compute min_new_max among those that do not increase.
        # But for simplicity: we first check if all increase. If yes, return None.
        # Otherwise, among all customers (including those increasing), we choose the one that minimizes new_max.
        # But condition says: if every available customer increases the new max beyond current max, return None.
        # So we need to detect if there exists at least one customer with new_max <= current_max.

        # We'll store the best among all customers (ignoring the increase condition) and also track if there is any non-increasing.
        if new_max < best_new_max or (new_max == best_new_max and new_return < best_new_return):
            best_new_max = new_max
            best_new_return = new_return
            best_idx = i

    # Now check if every customer increases the new max.
    all_increase = True
    for customer in available_customers:
        d_truck_cust = dist(current_position, customer)
        d_cust_depot = dist(customer, depot_position)
        new_return = d_truck_cust + d_cust_depot
        new_max = max(new_return, other_max)
        if new_max <= current_max:
            all_increase = False
            break

    if all_increase:
        return None
    else:
        return best_idx