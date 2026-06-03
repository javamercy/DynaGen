import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    n_cust = available_customers.shape[0]
    if n_cust == 0:
        return None
    # Identify active truck index
    dists_to_trucks = np.linalg.norm(truck_positions - current_position, axis=1)
    active_idx = np.argmin(dists_to_trucks)
    best_idx = None
    best_dist = float('inf')
    for i, cust in enumerate(available_customers):
        dists = np.linalg.norm(truck_positions - cust, axis=1)
        min_dist = np.min(dists)
        # Check if active truck is a nearest truck (or tie with lower index)
        if np.isclose(dists[active_idx], min_dist):
            # To break ties, ensure active_idx is the smallest index among those with min_dist
            if active_idx == np.argmin(dists):
                dist_current = np.linalg.norm(current_position - cust)
                if dist_current < best_dist:
                    best_dist = dist_current
                    best_idx = i
    return best_idx