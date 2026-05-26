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

    # own distances
    dist_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    own_cost = dist_to_cust + cust_to_depot

    # identify other trucks
    mask = np.all(np.abs(truck_positions - current_position) < 1e-8, axis=1)
    other_positions = truck_positions[~mask]

    if len(other_positions) == 0:
        # single truck: just minimize cost
        best_idx = int(np.argmin(own_cost))
        return best_idx

    # compute best other cost for each customer
    # other_to_cust: (n_customers, n_other)
    other_to_cust = np.linalg.norm(
        available_customers[:, None, :] - other_positions[None, :, :], axis=2
    )
    best_other_cost = np.min(other_to_cust, axis=1) + cust_to_depot
    regret = best_other_cost - own_cost

    max_regret = np.max(regret)
    if max_regret > 1e-6:
        return int(np.argmax(regret))

    # no positive regret: decide to wait or pick smallest cost
    current_depot_dist = np.linalg.norm(current_position - depot_position)
    other_depot_dists = np.linalg.norm(other_positions - depot_position, axis=1)
    max_other_depot_dist = np.max(other_depot_dists) if len(other_depot_dists) > 0 else -np.inf
    is_farthest = current_depot_dist >= max_other_depot_dist

    if not is_farthest and len(available_customers) > 2:
        return None  # wait
    else:
        return int(np.argmin(own_cost))