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

    # Compute centroid of available customers
    centroid = np.mean(available_customers, axis=0)

    # distances from current truck to each customer and customer to depot
    cur_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    cost_now = cur_to_cust + cust_to_depot

    # other trucks
    mask = np.all(np.abs(truck_positions - current_position) < 1e-8, axis=1)
    other_trucks = truck_positions[~mask]

    if len(other_trucks) == 0:
        best_idx = np.argmin(cost_now)
        return int(best_idx)

    # Best other cost for each customer
    other_to_cust = np.linalg.norm(
        available_customers[:, None, :] - other_trucks[None, :, :], axis=2
    )
    best_other_cost = np.min(other_to_cust, axis=1) + cust_to_depot

    regret = best_other_cost - cost_now

    # Check waiting condition: current truck is closest to depot and few customers
    current_depot_dist = np.linalg.norm(current_position - depot_position)
    other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
    is_closest_to_depot = current_depot_dist < np.min(other_depot_dists) - 1e-8

    if is_closest_to_depot and len(available_customers) <= len(other_trucks):
        return None

    # Compute scores with bonuses
    dist_to_centroid = np.linalg.norm(available_customers - centroid, axis=1)
    # Bonuses (negative penalties: we want to maximize score, so subtract distance)
    alpha = 0.3  # centroid weight
    beta = 0.2   # depot weight
    scores = regret + alpha * (-dist_to_centroid) + beta * (-cust_to_depot)

    best_idx = np.argmax(scores)
    return int(best_idx)