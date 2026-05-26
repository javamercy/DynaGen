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

    # farthest truck identification
    current_depot_dist = np.linalg.norm(current_position - depot_position)
    other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
    max_other_depot_dist = np.max(other_depot_dists) if len(other_depot_dists) > 0 else 0.0
    is_farthest = current_depot_dist >= max_other_depot_dist

    max_regret = np.max(regret)
    if max_regret > 1e-6:
        best_idx = np.argmax(regret)
        return int(best_idx)

    # no positive regret: decide to wait or take a customer
    if is_farthest:
        # farthest truck: take the best cost customer
        best_idx = np.argmin(cost_now)
        return int(best_idx)
    else:
        # not farthest: wait for more customers
        return None