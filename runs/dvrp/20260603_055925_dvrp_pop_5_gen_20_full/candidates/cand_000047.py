import numpy as np

def choose_next_customer(current_position: np.ndarray, depot_position: np.ndarray, truck_positions: np.ndarray, available_customers: np.ndarray) -> int | None:
    if available_customers.shape[0] == 0:
        return None

    n_trucks = truck_positions.shape[0]
    # single truck: always dispatch to minimize round-trip
    if n_trucks == 1:
        cust_dist = np.linalg.norm(available_customers - current_position, axis=1)
        ret_dist = np.linalg.norm(available_customers - depot_position, axis=1)
        own_round = cust_dist + ret_dist
        return int(np.argmin(own_round))

    # identify active truck index
    match = np.all(np.isclose(truck_positions, current_position), axis=1)
    if np.any(match):
        active_idx = np.where(match)[0][0]
    else:
        # fallback: closest truck
        dist_to_trucks = np.linalg.norm(truck_positions - current_position, axis=1)
        active_idx = np.argmin(dist_to_trucks)

    # distances to depot for all trucks
    truck_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # other trucks' max
    other_mask = np.ones(n_trucks, dtype=bool)
    other_mask[active_idx] = False
    if np.any(other_mask):
        other_max = np.max(truck_depot[other_mask])
    else:
        other_max = 0.0

    d_self = np.linalg.norm(current_position - depot_position)
    cust_dist = np.linalg.norm(available_customers - current_position, axis=1)
    ret_dist = np.linalg.norm(available_customers - depot_position, axis=1)
    own_round = cust_dist + ret_dist
    min_own = np.min(own_round)
    best_idx = np.argmin(own_round)

    waiting_ttt = max(d_self, other_max)

    # if at depot (d_self near 0), always dispatch
    if d_self < 1e-6:
        return int(best_idx)
    elif min_own <= waiting_ttt:
        return int(best_idx)
    else:
        return None