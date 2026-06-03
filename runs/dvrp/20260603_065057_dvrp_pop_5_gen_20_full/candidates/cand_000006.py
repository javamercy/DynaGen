import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None

    # identify the current truck's index in truck_positions
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diff)

    n_trucks = truck_positions.shape[0]
    n_cust = available_customers.shape[0]

    # distances from current truck to each customer
    d_curr = np.linalg.norm(available_customers - current_position, axis=1)
    # distances from each customer to depot
    d_depot = np.linalg.norm(available_customers - depot_position, axis=1)

    # compute minimum distance from other trucks to each customer
    if n_trucks > 1:
        # distances from all trucks to all customers
        # shape (n_trucks, n_cust)
        all_dists = np.linalg.norm(truck_positions[:, np.newaxis, :] - available_customers, axis=2)
        # set current truck's distances to infinity so they are not considered
        all_dists[current_idx, :] = np.inf
        d_other_min = np.min(all_dists, axis=0)
    else:
        # only one truck; set to a large value
        d_other_min = np.full(n_cust, 1e9)

    # compute score
    epsilon = 1e-6
    score = d_curr * d_depot / (d_other_min + epsilon)

    # always return the customer with minimum score (no wait)
    best_idx = np.argmin(score)
    return int(best_idx)