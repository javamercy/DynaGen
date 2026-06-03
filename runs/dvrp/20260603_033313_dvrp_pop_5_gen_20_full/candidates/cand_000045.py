import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    n_cust = available_customers.shape[0]
    if n_cust == 0:
        return None
    # Identify active truck index
    dists_to_current = np.linalg.norm(truck_positions - current_position, axis=1)
    active_idx = np.argmin(dists_to_current)
    # Precompute depot distances for all trucks
    depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    best_idx = None
    best_score = float('inf')
    for i, cust in enumerate(available_customers):
        # Distances from all trucks to customer
        dists_to_cust = np.linalg.norm(truck_positions - cust, axis=1)
        min_dist = np.min(dists_to_cust)
        # Check if active truck is unique nearest (or smallest index tie-break)
        if not np.isclose(dists_to_cust[active_idx], min_dist):
            continue
        if active_idx != np.argmin(dists_to_cust):
            continue
        # Compute score: estimated return distance for active truck
        score = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        if score < best_score:
            best_score = score
            best_idx = i
    return best_idx