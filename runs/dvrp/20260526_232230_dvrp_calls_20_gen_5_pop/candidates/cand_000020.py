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

    n_trucks = len(truck_positions)
    if n_trucks > 1:
        all_dist = dist(truck_positions, depot_position)
        idx = np.where(np.all(truck_positions == current_position, axis=1))[0]
        if len(idx) > 0:
            current_idx = idx[0]
            mask = np.ones(n_trucks, dtype=bool)
            mask[current_idx] = False
            other_returns = all_dist[mask]
            other_max = np.max(other_returns) if len(other_returns) > 0 else 0.0
        else:
            other_max = np.max(all_dist)
    else:
        other_max = None

    best_idx = None
    best_new_max = float('inf')
    best_new_return = float('inf')

    for i, customer in enumerate(available_customers):
        d_truck_cust = dist(current_position, customer)
        d_cust_depot = dist(customer, depot_position)
        new_return = d_truck_cust + d_cust_depot
        if other_max is not None:
            new_max = max(new_return, other_max)
        else:
            new_max = new_return
        if (new_max < best_new_max) or (new_max == best_new_max and new_return < best_new_return):
            best_new_max = new_max
            best_new_return = new_return
            best_idx = i

    if other_max is not None and best_new_max > other_max:
        return None
    return best_idx