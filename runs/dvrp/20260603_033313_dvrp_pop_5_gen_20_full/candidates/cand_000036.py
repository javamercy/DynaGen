import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    # distances from active truck to each customer
    active_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    # distances from each customer to depot
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)

    # for each customer, compute the distance from the closest other truck
    all_truck_dists = np.linalg.norm(
        available_customers[:, np.newaxis, :] - truck_positions[np.newaxis, :, :], axis=2
    )
    # identify which truck is active
    active_idx = np.where(np.all(truck_positions == current_position, axis=1))[0][0]
    # distance from other trucks: set active column to inf
    other_dists = all_truck_dists.copy()
    other_dists[:, active_idx] = np.inf
    min_other_dists = np.min(other_dists, axis=1)

    # customers where active is strictly the closest (if tie, active still considered closest? use <=)
    # To be conservative, use <= to break ties in favor of active
    active_is_closest = all_truck_dists[:, active_idx] <= min_other_dists

    # compute score = active_to_cust + cust_to_depot
    scores = active_to_cust + cust_to_depot

    if np.any(active_is_closest):
        # among eligible, pick the one with smallest score
        eligible_scores = scores[active_is_closest]
        best_eligible_idx = np.where(active_is_closest)[0][np.argmin(eligible_scores)]
        return int(best_eligible_idx)
    else:
        # fallback: pick overall smallest score
        best_idx = np.argmin(scores)
        return int(best_idx)