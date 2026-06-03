import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    n_cust = available_customers.shape[0]
    if n_cust == 0:
        return None
    # Identify active truck index
    dist_to_trucks = np.linalg.norm(truck_positions - current_position, axis=1)
    active_idx = np.argmin(dist_to_trucks)
    best_idx = None
    best_score = float('inf')
    for i, cust in enumerate(available_customers):
        dists = np.linalg.norm(truck_positions - cust, axis=1)
        min_dist = np.min(dists)
        # Check if active truck is a nearest (with tolerance)
        if np.isclose(dists[active_idx], min_dist):
            # Ensure active_idx is the smallest index among those with min_dist
            if active_idx == np.argmin(dists):
                # Score: travel to customer + return to depot
                score = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
                if score < best_score:
                    best_score = score
                    best_idx = i
    return best_idx