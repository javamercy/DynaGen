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

    # distances for current truck
    curr_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    cost_now = curr_to_cust + cust_to_depot

    # identify other trucks
    mask = np.all(np.abs(truck_positions - current_position) < 1e-8, axis=1)
    other_trucks = truck_positions[~mask]

    if len(other_trucks) == 0:
        # single truck: just minimize cost
        best_idx = np.argmin(cost_now)
        return int(best_idx)

    # regret calculation
    other_to_cust = np.linalg.norm(available_customers[:, None, :] - other_trucks[None, :, :], axis=2)
    best_other_cost = np.min(other_to_cust, axis=1) + cust_to_depot
    regret = best_other_cost - cost_now

    # bonus for farthest truck to reduce max depot distance
    current_depot_dist = np.linalg.norm(current_position - depot_position)
    other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
    max_depot_dist = np.max(other_depot_dists) if len(other_depot_dists) > 0 else current_depot_dist
    is_farthest = current_depot_dist >= max_depot_dist  # tie considered farthest
    if is_farthest:
        # add small bonus to all regrets to encourage taking customers
        regret += 0.1

    max_regret = np.max(regret)
    if max_regret > 1e-6:
        best_idx = np.argmax(regret)
        return int(best_idx)

    # no positive regret: decide whether to wait
    if len(available_customers) > 2 and not is_farthest:
        # wait because other trucks are farther and enough customers remain
        return None
    else:
        # otherwise, pick customer with minimal cost
        best_idx = np.argmin(cost_now)
        return int(best_idx)