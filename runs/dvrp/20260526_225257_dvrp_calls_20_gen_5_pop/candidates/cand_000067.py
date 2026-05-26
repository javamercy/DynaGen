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

    centroid = np.mean(available_customers, axis=0)

    cur_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    cost_now = cur_to_cust + cust_to_depot

    mask = np.all(np.abs(truck_positions - current_position) < 1e-8, axis=1)
    other_trucks = truck_positions[~mask]

    if len(other_trucks) == 0:
        best_idx = np.argmin(cost_now)
        return int(best_idx)

    other_to_cust = np.linalg.norm(
        available_customers[:, None, :] - other_trucks[None, :, :], axis=2
    )
    best_other_cost = np.min(other_to_cust, axis=1) + cust_to_depot
    regret = best_other_cost - cost_now

    current_depot_dist = np.linalg.norm(current_position - depot_position)
    other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
    is_farthest_from_depot = current_depot_dist > np.max(other_depot_dists) - 1e-8

    if len(available_customers) > 3 and np.max(regret) <= 0 and not is_farthest_from_depot:
        return None

    dist_to_centroid = np.linalg.norm(available_customers - centroid, axis=1)
    alpha = 0.2
    beta = 0.1
    scores = regret + alpha * (-dist_to_centroid) + beta * (-cust_to_depot)

    best_idx = np.argmax(scores)
    return int(best_idx)