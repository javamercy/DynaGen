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

    # Euclidean distances
    def dist(a, b):
        return np.linalg.norm(a - b, axis=-1)

    current_pos = current_position
    depot = depot_position
    other_trucks = truck_positions
    # Current return time for this truck if it returns now
    current_truck_return = dist(current_pos, depot)
    # Current max return time among other trucks
    other_returns = dist(other_trucks, depot)
    current_max_return = max(current_truck_return, np.max(other_returns))

    best_idx = None
    best_new_max = float('inf')
    best_new_return = float('inf')

    for i, customer in enumerate(available_customers):
        d_truck_cust = dist(current_pos, customer)
        d_cust_depot = dist(customer, depot)
        new_return = d_truck_cust + d_cust_depot
        # Compute new max return if this truck serves this customer
        # Other trucks remain as is
        new_max = max(new_return, np.max(other_returns))
        # Among all candidates, we want to minimize new_max
        # If tie, minimize new_return
        if new_max < best_new_max or (new_max == best_new_max and new_return < best_new_return):
            best_new_max = new_max
            best_new_return = new_return
            best_idx = i

    # After selection, we could consider waiting if best_new_max is significantly larger than current_max?
    # But we'll always serve the best candidate.
    return best_idx