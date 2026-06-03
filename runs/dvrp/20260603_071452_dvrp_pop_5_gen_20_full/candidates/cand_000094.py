import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None

    n_trucks = truck_positions.shape[0]
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    cur_to_cust = np.linalg.norm(available_customers - current_position, axis=1)

    if n_trucks == 1:
        active_cost = cur_to_cust + cust_to_depot
        best_idx = np.argmin(active_cost)
        return int(best_idx)

    # Find active truck index
    dist_to_pos = np.linalg.norm(truck_positions - current_position, axis=1)
    active_idx = np.argmin(dist_to_pos)

    # Vectorized costs: (n_trucks, n_avail)
    truck_cust_dists = np.linalg.norm(
        truck_positions[:, np.newaxis, :] - available_customers[np.newaxis, :, :], axis=2
    )
    costs = truck_cust_dists + cust_to_depot[np.newaxis, :]

    active_costs = costs[active_idx, :]
    mask = np.ones(n_trucks, dtype=bool)
    mask[active_idx] = False
    other_costs = costs[mask, :]
    other_min = np.min(other_costs, axis=0)

    scores = other_min - active_costs

    # Tie-break: lower active cost better
    best_idx = np.lexsort((active_costs, -scores))[0]

    if scores[best_idx] > 0:
        return int(best_idx)
    else:
        return None