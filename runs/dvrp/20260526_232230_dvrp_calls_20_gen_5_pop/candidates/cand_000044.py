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

    # Find the index of the current truck in truck_positions
    current_idx = np.where(np.all(truck_positions == current_position, axis=1))[0][0]

    # All trucks' current distances to depot
    all_returns = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max = np.max(all_returns)

    # Other trucks' returns (exclude current truck)
    other_returns = np.delete(all_returns, current_idx)
    if other_returns.shape[0] > 0:
        max_other = np.max(other_returns)
    else:
        max_other = -np.inf  # no other trucks

    best_idx = None
    best_new_max = float('inf')
    best_new_return = float('inf')

    for i, customer in enumerate(available_customers):
        d_truck_cust = np.linalg.norm(current_position - customer)
        d_cust_depot = np.linalg.norm(customer - depot_position)
        new_return = d_truck_cust + d_cust_depot
        new_max = max(new_return, max_other)

        # Greedy minimization of new max, tie-break by own return time
        if (new_max < best_new_max) or (new_max == best_new_max and new_return < best_new_return):
            best_new_max = new_max
            best_new_return = new_return
            best_idx = i

    # Waiting decision: if best new max exceeds current max by >10% and there are other trucks, wait.
    threshold = 0.1
    if other_returns.shape[0] > 0 and best_new_max > current_max * (1 + threshold):
        return None

    return best_idx