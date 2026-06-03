import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    # distances from active truck to customers
    dist_active = np.linalg.norm(available_customers - current_position, axis=1)
    # distances from customers to depot
    dist_depot = np.linalg.norm(available_customers - depot_position, axis=1)

    # find index of active truck in truck_positions (exact match)
    active_idx = np.where(np.all(truck_positions == current_position, axis=1))[0][0]

    # distances from all trucks to all customers: shape (n_cust, n_trucks)
    all_dists = np.linalg.norm(
        available_customers[:, np.newaxis, :] - truck_positions[np.newaxis, :, :], axis=2
    )

    # minimum distance from any other truck to each customer
    other_dists = np.copy(all_dists)
    other_dists[:, active_idx] = np.inf
    min_other = np.min(other_dists, axis=1)

    # active is closest (or tied) to customer?
    active_closest = all_dists[:, active_idx] <= min_other

    if np.any(active_closest):
        candidates = np.where(active_closest)[0]
        # priority = depot distance - active distance (encourages far depot but not too far from active)
        priorities = dist_depot[candidates] - dist_active[candidates]
        best_idx = candidates[np.argmax(priorities)]
    else:
        # fallback: nearest customer
        best_idx = np.argmin(dist_active)

    return int(best_idx)