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
    other_returns = dist(truck_positions, depot)
    other_max = np.max(other_returns)

    best_idx = None
    best_new_max = float('inf')
    best_new_return = float('inf')
    best_min_dist = float('inf')

    for i, customer in enumerate(available_customers):
        d_truck_cust = dist(current_pos, customer)
        d_cust_depot = dist(customer, depot)
        new_return = d_truck_cust + d_cust_depot
        new_max = max(new_return, other_max)

        # compute min distance to other available customers
        # mask out self
        others = np.delete(available_customers, i, axis=0)
        if others.shape[0] > 0:
            min_dist_to_others = np.min(dist(customer, others))
        else:
            min_dist_to_others = 0.0  # no others, not relevant

        # compare: primary new_max, secondary new_return, tertiary min_dist_to_others
        if (new_max < best_new_max or
            (new_max == best_new_max and new_return < best_new_return) or
            (new_max == best_new_max and new_return == best_new_return and min_dist_to_others < best_min_dist)):
            best_new_max = new_max
            best_new_return = new_return
            best_min_dist = min_dist_to_others
            best_idx = i

    return best_idx