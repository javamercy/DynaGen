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

    # Find index of current truck in truck_positions
    idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    all_returns = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_return = all_returns[idx]
    other_max = np.max(np.delete(all_returns, idx)) if len(all_returns) > 1 else -np.inf
    current_max_if_depot = max(current_return, other_max)

    best_idx = None
    best_new_max = float('inf')
    best_new_return = float('inf')

    for i, customer in enumerate(available_customers):
        new_return = np.linalg.norm(current_position - customer) + np.linalg.norm(customer - depot_position)
        new_max = max(new_return, other_max)
        if new_max < best_new_max or (new_max == best_new_max and new_return < best_new_return):
            best_new_max = new_max
            best_new_return = new_return
            best_idx = i

    if best_new_max <= current_max_if_depot:
        return best_idx
    else:
        return None