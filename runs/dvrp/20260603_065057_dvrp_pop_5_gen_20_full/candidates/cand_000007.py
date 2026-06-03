import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None

    # Distances from current truck to customers
    d_current = np.linalg.norm(available_customers - current_position, axis=1)
    # Distances from customers to depot
    d_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    tour_current = d_current + d_to_depot

    # Compute best other truck tour time
    if len(truck_positions) == 0:
        best_other_tour = np.full(len(available_customers), np.inf)
    else:
        # Distances from each other truck to each customer
        diff = truck_positions[:, np.newaxis, :] - available_customers[np.newaxis, :, :]  # (n_trucks, n_cust, 2)
        dist_other = np.sqrt(np.sum(diff**2, axis=-1))  # (n_trucks, n_cust)
        other_tour = dist_other + d_to_depot[np.newaxis, :]  # (n_trucks, n_cust)
        best_other_tour = np.min(other_tour, axis=0)  # (n_cust,)

    regret = tour_current - best_other_tour

    # If all regrets > 0, wait
    if np.all(regret > 0):
        return None

    # Choose among customers with non-positive regret
    candidates = np.where(regret <= 0)[0]
    if len(candidates) == 0:
        return None

    # Compute density (KDE) for all customers
    n_cust = len(available_customers)
    if n_cust == 1:
        # Only one customer, density is zero
        densities = np.zeros(1)
    else:
        # Pairwise distances
        diff = available_customers[:, np.newaxis, :] - available_customers[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dist_matrix, np.inf)
        nearest_neighbor = np.min(dist_matrix, axis=1)
        bandwidth = np.median(nearest_neighbor)
        if bandwidth == 0 or np.isnan(bandwidth):
            bandwidth = 1.0
        # Gaussian kernel density
        kernel = np.exp(- (dist_matrix / bandwidth)**2)
        np.fill_diagonal(kernel, 0.0)
        densities = np.sum(kernel, axis=1)

    # Among candidates, find the one with smallest regret (most negative)
    min_regret = np.min(regret[candidates])
    best_mask = (regret == min_regret) & (regret <= 0)
    best_indices = np.where(best_mask)[0]
    if len(best_indices) == 0:
        return int(candidates[0])  # fallback
    # Tie-break by highest density
    best_idx = best_indices[np.argmax(densities[best_indices])]
    return int(best_idx)