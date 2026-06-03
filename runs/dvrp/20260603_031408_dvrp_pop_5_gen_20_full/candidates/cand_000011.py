import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    n_trucks = len(truck_positions)
    if n_trucks == 1:
        # fallback to nearest neighbor
        dists = np.linalg.norm(available_customers - current_position, axis=1)
        return int(np.argmin(dists))
    
    best_idx = None
    best_profit = -np.inf
    best_curr_dist = np.inf
    for i, cust in enumerate(available_customers):
        d_curr = np.linalg.norm(cust - current_position)
        # distances to all trucks
        dists_all = np.linalg.norm(truck_positions - cust, axis=1)
        # find index of current truck in truck_positions
        # We need to identify which truck is the current one by position
        current_mask = np.allclose(truck_positions, current_position, atol=1e-8)
        if np.any(current_mask):
            current_idx = np.where(current_mask)[0][0]
            # distances to other trucks
            mask = np.ones(n_trucks, dtype=bool)
            mask[current_idx] = False
            d_other = np.min(dists_all[mask])
        else:
            # current_position not exactly in truck_positions? fallback: min over all
            d_other = np.min(dists_all)
        profit = d_other / (d_curr + 1e-8)
        if profit > best_profit or (profit == best_profit and d_curr < best_curr_dist):
            best_profit = profit
            best_curr_dist = d_curr
            best_idx = i
    return best_idx