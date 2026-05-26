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
    # Identify index of current truck
    dist_to_trucks = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = int(np.argmin(dist_to_trucks))
    other_indices = [i for i in range(len(truck_positions)) if i != current_idx]
    other_trucks = truck_positions[other_indices]
    
    # Waiting condition: if current truck is much farther than others from all customers
    if len(other_trucks) > 0:
        cur_min_dist = np.min(np.linalg.norm(available_customers - current_position, axis=1))
        other_min_dists = [
            np.min(np.linalg.norm(available_customers - p, axis=1))
            for p in other_trucks
        ]
        avg_other_min = np.mean(other_min_dists)
        if avg_other_min > 0 and cur_min_dist > 1.5 * avg_other_min:
            return None
    
    # Compute distances from current truck to each customer
    d_cur = np.linalg.norm(available_customers - current_position, axis=1)
    # Compute distances from each customer to depot
    d_dep = np.linalg.norm(available_customers - depot_position, axis=1)
    # Compute for each customer the distance to the nearest other truck
    if len(other_trucks) > 0:
        # other_trucks shape (n_other, 2), customers (n_cust, 2) -> distances (n_other, n_cust)
        dist_other = np.linalg.norm(other_trucks[:, np.newaxis, :] - available_customers[np.newaxis, :, :], axis=2)
        nearest_other = np.min(dist_other, axis=0)
    else:
        nearest_other = np.zeros(len(available_customers))  # no other trucks, no penalty
    
    # Compute fleet centroid and current truck's distance to it
    centroid = np.mean(truck_positions, axis=0)
    centroid_dist = np.linalg.norm(current_position - centroid)
    # Normalize centroid_dist by max distance of any truck to centroid
    max_centroid_dist = np.max(np.linalg.norm(truck_positions - centroid, axis=1))
    if max_centroid_dist > 0:
        normalized_centroid_dist = centroid_dist / max_centroid_dist
    else:
        normalized_centroid_dist = 0.0
    # Dynamic depot coefficient: base 0.3 + 0.3 * normalized_centroid_dist
    alpha = 0.3 + 0.3 * normalized_centroid_dist
    gamma = 0.4  # weight for nearest_other penalty
    # Score: lower is better. Include -gamma*nearest_other to favor customers far from other trucks
    score = d_cur + alpha * d_dep - gamma * nearest_other
    best_idx = int(np.argmin(score))
    return best_idx