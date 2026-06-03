import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    
    # Calculate centroid of available customers
    centroid = np.mean(available_customers, axis=0)
    
    # Distances from current position to each customer
    dist_current = np.linalg.norm(available_customers - current_position, axis=1)
    # Distances from centroid to each customer
    dist_centroid = np.linalg.norm(available_customers - centroid, axis=1)
    
    # Compute distance-related adaptive factors
    n_customers = len(available_customers)
    avg_truck_dist = np.mean(np.linalg.norm(truck_positions - depot_position, axis=1))
    my_dist_to_depot = np.linalg.norm(current_position - depot_position)
    
    # Factor based on remaining customer count (0 to 1, lower when many customers)
    customer_factor = 1 - 0.5 * min(n_customers / 100, 1.0)
    # Factor based on truck's relative distance to depot (0 to 1, lower when far)
    dist_factor = 1 - 0.5 * min(my_dist_to_depot / (avg_truck_dist + 1e-6), 1.0)
    
    alpha = 0.5 * customer_factor * dist_factor
    alpha = max(0.1, min(0.5, alpha))  # clamp between 0.1 and 0.5
    
    scores = dist_current + alpha * dist_centroid
    best_idx = int(np.argmin(scores))
    return best_idx