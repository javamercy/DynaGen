import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    n_trucks = truck_positions.shape[0]
    active_index = np.where(np.all(truck_positions == current_position, axis=1))[0][0]

    # single truck: serve nearest customer
    if n_trucks == 1:
        dist_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
        return int(np.argmin(dist_to_cust))

    # distances to depot
    active_dist_depot = np.linalg.norm(current_position - depot_position)
    truck_dists_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    max_other_dist = np.max(np.delete(truck_dists_depot, active_index))

    # wait if active is farthest or tied
    if active_dist_depot >= max_other_dist:
        return None

    # distances for each customer
    dist_active_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    dist_cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    T_active = dist_active_to_cust + dist_cust_to_depot

    # feasible customers that do not increase max
    feasible = T_active <= max_other_dist + 1e-9
    if not np.any(feasible):
        return None

    # compute closest truck per customer
    all_truck_dists = np.linalg.norm(
        available_customers[:, np.newaxis, :] - truck_positions[np.newaxis, :, :], axis=2
    )
    other_dists = all_truck_dists.copy()
    other_dists[:, active_index] = np.inf
    min_other_dists = np.min(other_dists, axis=1)

    active_closest = (all_truck_dists[:, active_index] <= min_other_dists) & feasible

    if np.any(active_closest):
        candidates = np.where(active_closest)[0]
        best_idx = candidates[np.argmax(dist_cust_to_depot[candidates])]
    else:
        candidates = np.where(feasible)[0]
        best_idx = candidates[np.argmin(dist_active_to_cust[candidates])]

    return int(best_idx)