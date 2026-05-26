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

    # Compute distances from all trucks to depot
    truck_depot_dists = dist(truck_positions, depot_position)

    # Identify current truck index
    dists_to_current = dist(truck_positions, current_position)
    current_idx = np.argmin(dists_to_current)

    current_truck_return = truck_depot_dists[current_idx]

    # Exclude current truck from other trucks
    other_dists = np.delete(truck_depot_dists, current_idx)
    other_max = np.max(other_dists) if other_dists.size > 0 else -np.inf
    current_max = max(current_truck_return, other_max)

    best_idx = None
    best_new_max = float('inf')
    best_new_return = float('inf')

    for i, customer in enumerate(available_customers):
        d_truck_cust = dist(current_position, customer)
        d_cust_depot = dist(customer, depot_position)
        new_return = d_truck_cust + d_cust_depot
        new_max = max(new_return, other_max)

        if best_idx is None or new_max < best_new_max or (new_max == best_new_max and new_return < best_new_return):
            best_new_max = new_max
            best_new_return = new_return
            best_idx = i

    # Wait if all customers increase max
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

    return best_idx