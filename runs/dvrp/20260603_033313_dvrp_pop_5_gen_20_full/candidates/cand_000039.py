import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    # Distances
    dist_active_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    dist_cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)

    # Active truck index
    active_idx = np.where(np.all(truck_positions == current_position, axis=1))[0][0]

    # All trucks to all customers
    all_dists = np.linalg.norm(
        available_customers[:, np.newaxis, :] - truck_positions[np.newaxis, :, :], axis=2
    )
    # Min distance from other trucks
    other_dists = np.copy(all_dists)
    other_dists[:, active_idx] = np.inf
    min_other_dists = np.min(other_dists, axis=1)

    active_is_closest = all_dists[:, active_idx] <= min_other_dists
    num_closest = np.sum(active_is_closest)

    # Active truck's distance to depot
    active_depot_dist = np.linalg.norm(current_position - depot_position)
    median_depot_dist = np.median(np.linalg.norm(truck_positions - depot_position, axis=1))

    # Waiting condition from parent
    if active_depot_dist < median_depot_dist and num_closest == 0:
        return None

    # Other trucks' current distances to depot
    other_truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    # For active, we consider it's at current_position before moving

    if num_closest > 0:
        candidate_indices = np.where(active_is_closest)[0]
        # Compute estimated max return time after serving each candidate
        best_idx = None
        best_max_return = np.inf
        for idx in candidate_indices:
            active_return = dist_active_to_cust[idx] + dist_cust_to_depot[idx]
            # Consider other trucks' current depot distances (they haven't moved)
            max_return = max(np.max(other_truck_depot_dists), active_return)
            # Tie-break: prefer smaller distance from active? Or larger depot distance?
            if max_return < best_max_return:
                best_max_return = max_return
                best_idx = idx
            elif max_return == best_max_return:
                # Tie: choose farthest from depot (as parent did)
                if dist_cust_to_depot[idx] > dist_cust_to_depot[best_idx]:
                    best_idx = idx
        return int(best_idx)
    else:
        # No closest customer: pick nearest customer
        best_idx = np.argmin(dist_active_to_cust)
        return int(best_idx)