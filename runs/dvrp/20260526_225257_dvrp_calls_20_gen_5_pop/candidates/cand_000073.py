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

    # distances from current truck to each customer
    curr_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    # distances from each customer to depot
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
    other_to_cust = np.linalg.norm(
        available_customers[:, None, :] - other_trucks[None, :, :], axis=2
    )
    best_other_cost = np.min(other_to_cust, axis=1) + cust_to_depot
    regret = best_other_cost - cost_now

    max_regret = np.max(regret)
    if max_regret > 1e-6:
        # urgent customer exists
        best_idx = np.argmax(regret)
        return int(best_idx)
    else:
        # no urgent customer: decide whether to wait
        # check if few customers remain
        if len(available_customers) <= 3:
            # force dispatch to avoid idle time
            # use score with spread bonus
            nearest_other = np.min(other_to_cust, axis=1)  # distance to nearest other truck
            score = cost_now - 0.1 * nearest_other  # bonus for being far from other trucks
            best_idx = np.argmin(score)
            return int(best_idx)

        current_depot_dist = np.linalg.norm(current_position - depot_position)
        other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
        max_other_depot = np.max(other_depot_dists)
        is_farthest = current_depot_dist >= max_other_depot - 1e-6

        if not is_farthest:
            # another truck is farther from depot; wait here
            return None
        else:
            # current truck is farthest or tied; dispatch with spread bonus
            nearest_other = np.min(other_to_cust, axis=1)
            score = cost_now - 0.1 * nearest_other
            best_idx = np.argmin(score)
            return int(best_idx)