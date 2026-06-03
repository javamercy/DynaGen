import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    n_cust = available_customers.shape[0]
    if n_cust == 0:
        return None
    # active truck index (the one deciding)
    dists_to_trucks = np.linalg.norm(truck_positions - current_position, axis=1)
    active_idx = np.argmin(dists_to_trucks)
    best_idx = None
    best_val = -float('inf')  # maximize distance to nearest other truck
    for i, cust in enumerate(available_customers):
        dists = np.linalg.norm(truck_positions - cust, axis=1)
        min_dist = np.min(dists)
        # eligibility: active truck must be a nearest truck (with tie-break by index)
        if np.isclose(dists[active_idx], min_dist) and active_idx == np.argmin(dists):
            # distance to nearest other truck (excluding active)
            other_dists = np.delete(dists, active_idx)
            min_other = np.min(other_dists) if len(other_dists) > 0 else 0.0
            if min_other > best_val:
                best_val = min_other
                best_idx = i
    return best_idx