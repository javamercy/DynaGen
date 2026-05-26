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
    
    mask = np.all(truck_positions == current_position, axis=1)
    if mask.any():
        other_trucks = truck_positions[~mask]
    else:
        other_trucks = truck_positions
    
    # Compute distances from each truck to customers for waiting condition
    if len(other_trucks) > 0:
        # For each other truck, compute min distance to any available customer
        other_min_dists = []
        for truck in other_trucks:
            d = np.linalg.norm(available_customers - truck, axis=1)
            other_min_dists.append(np.min(d))
        avg_other_min = np.mean(other_min_dists)
        
        # Current truck's min distance to customers
        current_min = np.min(dist_current)
        
        # Distance of current truck to depot
        truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
        mean_truck_depot_dist = np.mean(truck_depot_dists)
        current_to_depot = np.linalg.norm(current_position - depot_position)
        
        # Wait condition: current truck is far from customers compared to others, and not too close to depot
        if current_min > 1.2 * avg_other_min and current_to_depot > 0.4 * mean_truck_depot_dist:
            return None
    
    # Dynamic depot coefficient from parent
    centroid = np.mean(truck_positions, axis=0)
    truck_to_centroid = np.linalg.norm(current_position - centroid)
    mean_centroid_dist = np.mean(np.linalg.norm(truck_positions - centroid, axis=1))
    if mean_centroid_dist < 1e-6:
        depot_coefficient = 0.3
    else:
        depot_coefficient = 0.3 + 0.1 * (truck_to_centroid / mean_centroid_dist)
        depot_coefficient = min(depot_coefficient, 0.5)
    
    if len(other_trucks) > 0:
        diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
        dist_other = np.linalg.norm(diff, axis=2)
        nearest_other = np.min(dist_other, axis=1)
        score = dist_current - nearest_other + depot_coefficient * dist_depot
    else:
        score = dist_current + depot_coefficient * dist_depot
    
    best_idx = np.argmin(score)
    return int(best_idx)