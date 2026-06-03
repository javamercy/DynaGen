import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    # distances
    dist_active_to_customers = np.linalg.norm(available_customers - current_position, axis=1)
    dist_cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    active_to_depot = np.linalg.norm(current_position - depot_position)

    # find active truck index
    active_index = np.where(np.all(truck_positions == current_position, axis=1))[0][0]
    all_truck_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    other_depot_dists = np.delete(all_truck_depot, active_index)

    if other_depot_dists.size > 0:
        mean_other_depot = np.mean(other_depot_dists)
    else:
        mean_other_depot = active_to_depot

    # distances to all customers from all trucks
    all_truck_dists = np.linalg.norm(
        available_customers[:, np.newaxis, :] - truck_positions[np.newaxis, :, :], axis=2
    )
    other_dists = np.copy(all_truck_dists)
    other_dists[:, active_index] = np.inf
    min_other_dists = np.min(other_dists, axis=1)
    active_is_closest = all_truck_dists[:, active_index] <= min_other_dists

    if np.any(active_is_closest):
        candidates = np.where(active_is_closest)[0]
        if active_to_depot > mean_other_depot:
            # active is far: minimize total trip (dist to customer + customer to depot)
            total = dist_active_to_customers[candidates] + dist_cust_to_depot[candidates]
            best_idx = candidates[np.argmin(total)]
        else:
            # active is near: maximize customer distance to depot
            best_idx = candidates[np.argmax(dist_cust_to_depot[candidates])]
    else:
        # active is not closest to any customer
        if active_to_depot > mean_other_depot:
            total = dist_active_to_customers + dist_cust_to_depot
            best_idx = np.argmin(total)
        else:
            best_idx = np.argmax(dist_cust_to_depot)

    return int(best_idx)