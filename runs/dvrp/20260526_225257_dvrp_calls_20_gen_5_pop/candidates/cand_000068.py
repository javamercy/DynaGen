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

    # distance from current truck to each customer
    curr_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    # distance from each customer to depot
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    # cost if current truck serves customer and returns to depot
    cost_now = curr_to_cust + cust_to_depot

    # identify other trucks
    mask = np.all(np.abs(truck_positions - current_position) < 1e-8, axis=1)
    other_trucks = truck_positions[~mask]

    if len(other_trucks) == 0:
        # only one truck: always move to the customer with smallest cost
        best_idx = np.argmin(cost_now)
        return int(best_idx)

    # for each customer, compute best cost among other trucks
    # shape: (n_available, n_other)
    other_to_cust = np.linalg.norm(
        available_customers[:, None, :] - other_trucks[None, :, :], axis=2
    )
    # best other cost: min over other trucks of (dist(truck, customer) + customer to depot)
    best_other_cost = np.min(other_to_cust, axis=1) + cust_to_depot
    regret = best_other_cost - cost_now

    max_regret = np.max(regret)
    if max_regret > 1e-6:
        # urgent customer exists
        best_idx = np.argmax(regret)
        return int(best_idx)
    else:
        # no urgent customer: decide whether to wait
        current_depot_dist = np.linalg.norm(current_position - depot_position)
        other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
        max_other_depot = np.max(other_depot_dists)
        is_farthest = current_depot_dist >= max_other_depot - 1e-6

        if not is_farthest:
            # another truck is farther from depot; wait here
            return None
        else:
            # current truck is farthest or tied; move to customer with lowest cost (or highest regret, but regret <=0)
            # choose customer minimizing cost_now
            best_idx = np.argmin(cost_now)
            return int(best_idx)