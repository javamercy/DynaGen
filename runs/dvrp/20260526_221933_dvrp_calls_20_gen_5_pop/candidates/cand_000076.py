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
    
    # Identify other trucks (excluding current)
    mask = np.all(truck_positions == current_position, axis=1)
    if mask.any():
        other_trucks = truck_positions[~mask]
    else:
        other_trucks = truck_positions
    
    if len(other_trucks) == 0:
        score = dist_current + 0.3 * dist_depot
        best_idx = np.argmin(score)
        return int(best_idx)
    
    # Compute nearest other truck distance for each customer
    diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
    dist_other = np.linalg.norm(diff, axis=2)
    nearest_other = np.min(dist_other, axis=1)
    
    # Dynamic depot coefficient based on distance to centroid of available customers
    customer_centroid = np.mean(available_customers, axis=0)
    truck_to_centroid = np.linalg.norm(current_position - customer_centroid)
    all_trucks_to_centroid = np.linalg.norm(truck_positions - customer_centroid, axis=1)
    mean_centroid_dist = np.mean(all_trucks_to_centroid)
    if mean_centroid_dist < 1e-6:
        depot_coefficient = 0.3
    else:
        depot_coefficient = 0.2 + 0.2 * (truck_to_centroid / mean_centroid_dist)
        depot_coefficient = min(depot_coefficient, 0.5)
    
    # Wait condition: if current truck is much farther than others for all customers
    wait_threshold = 1.3
    if np.all(dist_current > wait_threshold * nearest_other):
        return None
    
    score = dist_current - nearest_other + depot_coefficient * dist_depot
    best_idx = np.argmin(score)
    return int(best_idx)