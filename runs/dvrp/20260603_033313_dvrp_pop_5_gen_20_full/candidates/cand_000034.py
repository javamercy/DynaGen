import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    # distances from active truck to customers
    d_active_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    # distances from customers to depot
    d_cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)

    # distances from all trucks to depot
    d_truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    active_depot_dist = np.linalg.norm(current_position - depot_position)
    avg_depot_dist = np.mean(d_truck_to_depot)
    active_is_underutilized = active_depot_dist < avg_depot_dist

    # find index of active truck in truck_positions
    active_index = np.where(np.all(truck_positions == current_position, axis=1))[0][0]

    # distances from all trucks to all customers: (n_cust, n_trucks)
    all_truck_dists = np.linalg.norm(
        available_customers[:, np.newaxis, :] - truck_positions[np.newaxis, :, :], axis=2
    )
    # minimum distance from other trucks to each customer
    other_dists = np.copy(all_truck_dists)
    other_dists[:, active_index] = np.inf
    min_other_dists = np.min(other_dists, axis=1)

    # active truck is closest (or tied) to customer?
    active_is_closest = all_truck_dists[:, active_index] <= min_other_dists

    if active_is_underutilized:
        # underutilized: push to far customers
        if np.any(active_is_closest):
            candidates = np.where(active_is_closest)[0]
            best_idx = candidates[np.argmax(d_cust_to_depot[candidates])]
        else:
            # no customer where active is closest, pick farthest overall
            best_idx = np.argmax(d_cust_to_depot)
    else:
        # overutilized (or equal): reduce route by going to nearest customer
        best_idx = np.argmin(d_active_to_cust)

    return int(best_idx)