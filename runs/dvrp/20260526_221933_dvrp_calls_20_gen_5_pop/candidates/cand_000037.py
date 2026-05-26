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
    
    c = current_position
    d = depot_position
    
    # distances from current truck to customers
    dist_to_cust = np.linalg.norm(available_customers - c, axis=1)
    # distances from customers to depot
    cust_to_depot = np.linalg.norm(available_customers - d, axis=1)
    
    # For each customer, find distance to nearest other truck (excluding current truck)
    # Create mask to exclude current truck from truck_positions
    # We need to compare each customer to all other truck positions
    # Truck_positions shape (n_trucks, 2). Current truck is at c, but its position is also in truck_positions.
    # We assume the current truck is in truck_positions, but we need to exclude it.
    # Approach: compute distances from each customer to all trucks, then find the minimum distance that is not zero (or not the current truck).
    # Since coordinates may be close, we need to identify which truck is the current one.
    # We can compute distance from each customer to current position c and to other trucks.
    # To find nearest other truck, we can compute distances to all trucks, then set the distance to current truck to infinity.
    
    # Compute distances from each customer to all trucks (n_available, n_trucks)
    dist_to_all_trucks = np.linalg.norm(
        available_customers[:, np.newaxis, :] - truck_positions[np.newaxis, :, :], axis=2
    )
    # Identify which truck is current: find the row index in truck_positions that matches c
    # We assume c is exactly one of the truck_positions (within floating tolerance). Use allclose.
    # Find index of current truck
    current_truck_idx = np.argmin(np.linalg.norm(truck_positions - c, axis=1))
    # Set distance to current truck to infinity to ignore it
    dist_to_all_trucks[:, current_truck_idx] = np.inf
    # Find nearest other truck distance for each customer
    min_dist_other = np.min(dist_to_all_trucks, axis=1)
    
    # Weights
    w_depot = 0.8
    w_other = 0.3
    eps = 1e-5
    
    # Compute score: lower is better
    # Add penalty for customers close to other trucks: penalty = w_other / (min_dist_other + eps)
    penalty_other = w_other / (min_dist_other + eps)
    score = dist_to_cust + w_depot * cust_to_depot + penalty_other
    
    # Tie-breaking: prefer farther customers (larger cust_to_depot) when scores equal
    score -= 1e-6 * cust_to_depot
    
    best_idx = int(np.argmin(score))
    return best_idx