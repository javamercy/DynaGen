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
    
    dist_current = np.linalg.norm(available_customers - current_position, axis=1)
    dist_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    
    mask = ~np.all(truck_positions == current_position, axis=1)
    other_trucks = truck_positions[mask]
    
    if len(other_trucks) == 0:
        # no other trucks, no waiting and no balancing
        nearest_other = np.zeros(len(available_customers))
        score = dist_current + 0.3 * dist_depot
        best_idx = np.argmin(score)
        return int(best_idx)
    
    # compute nearest other truck distance for each customer
    diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
    dist_other = np.linalg.norm(diff, axis=2)  # (n_available, n_other)
    nearest_other = np.min(dist_other, axis=1)
    
    # waiting condition: if current truck's closest customer is much farther than other's closest
    min_self = np.min(dist_current)
    min_other = np.min(nearest_other)
    if min_self > 1.5 * min_other:
        return None
    
    # dynamic depot coefficient based on distance from fleet centroid
    centroid = np.mean(truck_positions, axis=0)
    dist_to_centroid = np.linalg.norm(current_position - centroid)
    dists_to_centroid = np.linalg.norm(truck_positions - centroid, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist > 0:
        factor = dist_to_centroid / mean_dist
        depot_coef = 0.3 + 0.3 * min(factor, 2.0)  # cap at 0.9
    else:
        depot_coef = 0.3
    
    score = dist_current - nearest_other + depot_coef * dist_depot
    best_idx = np.argmin(score)
    return int(best_idx)