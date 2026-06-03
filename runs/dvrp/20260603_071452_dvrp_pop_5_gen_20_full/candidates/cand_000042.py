import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    n_trucks = len(truck_positions)
    # distances
    dist_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    centroid = np.mean(available_customers, axis=0)
    dist_to_centroid = np.linalg.norm(available_customers - centroid, axis=1)
    alpha = 0.3
    cost = dist_to_cust + cust_to_depot + alpha * dist_to_centroid
    # identify current truck
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    idx_current = int(np.argmin(diff))
    all_truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    if n_trucks == 1:
        # single truck: no waiting, just pick best cost
        best_idx = int(np.argmin(cost))
        return best_idx
    # multiple trucks
    other_max = np.max(np.delete(all_truck_depot_dists, idx_current))
    candidate_ttt = np.maximum(cost, other_max)
    min_ttt = np.min(candidate_ttt)
    # adaptive slack
    avg_other_dist = np.mean(np.delete(all_truck_depot_dists, idx_current))
    truck_depot_dist = all_truck_depot_dists[idx_current]
    base_slack = 0.05 / n_trucks
    if truck_depot_dist > avg_other_dist:
        slack = base_slack * 0.5
    else:
        slack = base_slack
    if min_ttt > other_max + slack:
        return None
    # pick best among those with min TTT
    tol = 1e-9
    mask = candidate_ttt <= min_ttt + tol
    filtered_dist = dist_to_cust[mask]
    filtered_centroid = dist_to_centroid[mask]
    tie_score = filtered_dist + 0.1 * filtered_centroid
    best_local_idx = int(np.argmin(tie_score))
    global_indices = np.where(mask)[0]
    best_idx = int(global_indices[best_local_idx])
    return best_idx