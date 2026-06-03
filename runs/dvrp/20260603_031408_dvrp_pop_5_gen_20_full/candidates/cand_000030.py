import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None

    # distances from active truck to each customer
    d_active_cust = np.linalg.norm(available_customers - current_position, axis=1)
    # distances from each customer to depot
    d_cust_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    # new return time for active truck if it serves customer i
    new_return = d_active_cust + d_cust_depot

    # distances from all trucks to depot
    truck_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # find index of active truck (the one at current_position)
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    # maximum of other trucks' current depot distances
    max_other = np.max(np.delete(truck_depot, active_idx))

    # overall maximum return time after serving each customer
    candidate_max = np.maximum(new_return, max_other)
    min_val = np.min(candidate_max)
    ties = np.where(candidate_max == min_val)[0]

    if len(ties) == 1:
        return int(ties[0])
    else:
        # tie-breaking: choose the customer with smallest distance from active truck
        best_idx = ties[np.argmin(d_active_cust[ties])]
        return int(best_idx)