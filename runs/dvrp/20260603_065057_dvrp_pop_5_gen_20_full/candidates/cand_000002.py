import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    
    n_cust = len(available_customers)
    
    # Compute pairwise distances among available customers
    # Use broadcasting to compute matrix of shape (n_cust, n_cust)
    diff = available_customers[:, np.newaxis, :] - available_customers[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))  # (n_cust, n_cust)
    
    # For each customer, find nearest neighbor distance (excluding self)
    # Fill diagonal with inf to ignore self
    np.fill_diagonal(dist_matrix, np.inf)
    nearest_neighbor_dists = np.min(dist_matrix, axis=1)
    # Bandwidth as median of nearest neighbor distances
    bandwidth = np.median(nearest_neighbor_dists)
    if bandwidth == 0:
        bandwidth = 1.0  # fallback if all customers coincident
    
    # Kernel density estimate for each customer
    # Gaussian kernel with bandwidth
    kernel_vals = np.exp(- (dist_matrix / bandwidth)**2)
    # Diagonal was inf, so exp(-inf) = 0, but reset self contribution to 0
    np.fill_diagonal(kernel_vals, 0.0)
    densities = np.sum(kernel_vals, axis=1)
    
    # Distance from current truck to each customer
    d_current = np.sqrt(np.sum((current_position - available_customers)**2, axis=1))
    
    # Distance to nearest other truck for each customer
    # Identify current truck index (position may have floating point representation)
    # Use np.allclose to avoid floating point issues
    truck_diff = truck_positions - current_position  # (n_trucks, 2)
    is_current = np.all(np.abs(truck_diff) < 1e-12, axis=1)  # tolerance
    # If multiple trucks at same position (unlikely), pick first
    current_idx = np.where(is_current)[0][0] if np.any(is_current) else None
    
    # Compute distances from customer to all trucks
    # Subtract current truck's distance if we know its index
    # Alternative: compute all distances and then exclude current truck
    truck_distances = np.sqrt(np.sum((truck_positions[np.newaxis, :, :] - available_customers[:, np.newaxis, :])**2, axis=-1))  # (n_cust, n_trucks)
    if current_idx is not None:
        # Set distance to current truck to infinity so it's not considered
        truck_distances[:, current_idx] = np.inf
    min_other_dist = np.min(truck_distances, axis=1)
    # If there is no other truck (n_trucks=1), min_other_dist will be inf; set to large number
    min_other_dist = np.where(np.isfinite(min_other_dist), min_other_dist, 1e6)
    
    # Score: density / (d_current * min_other_dist + epsilon)
    epsilon = 1e-8
    scores = densities / (d_current * min_other_dist + epsilon)
    
    # Handle case where all scores are zero (degenerate)
    if np.all(scores == 0):
        # Fallback: choose customer closest to current position
        return int(np.argmin(d_current))
    
    return int(np.argmax(scores))