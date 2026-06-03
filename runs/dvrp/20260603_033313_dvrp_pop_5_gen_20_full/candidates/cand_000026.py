import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    
    n_trucks = len(truck_positions)
    # Find index of current truck in truck_positions
    idx_current = None
    for j in range(n_trucks):
        if np.linalg.norm(truck_positions[j] - current_position) < 1e-9:
            idx_current = j
            break
    if idx_current is None:
        # fallback: assume first truck? shouldn't happen
        idx_current = 0
    
    # Precompute distances
    cust_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    truck_to_cust = np.zeros((n_trucks, len(available_customers)))
    for j in range(n_trucks):
        truck_to_cust[j] = np.linalg.norm(truck_positions[j] - available_customers, axis=1)
    
    # Current truck's direct return time
    current_return = np.linalg.norm(current_position - depot_position)
    
    # Best completion for each other truck (excluding current) considering all customers
    best_other_all = np.full(n_trucks, np.inf)
    for j in range(n_trucks):
        if j == idx_current:
            continue
        # option: direct return
        best = truck_to_depot[j]
        # option: serve any customer
        min_cust = np.min(truck_to_cust[j] + cust_depot)
        best = min(best, min_cust)
        best_other_all[j] = best
    max_other_all = np.max(best_other_all) if n_trucks > 1 else 0.0
    
    # Evaluate wait option
    max_wait = max(current_return, max_other_all)
    best_max = max_wait
    best_idx = None  # None for wait
    best_current = current_return
    
    # Evaluate each customer
    for i in range(len(available_customers)):
        cur_comp = np.linalg.norm(current_position - available_customers[i]) + cust_depot[i]
        # Compute for each other truck best completion excluding customer i
        best_other_excl = []
        for j in range(n_trucks):
            if j == idx_current:
                continue
            best = truck_to_depot[j]
            # min over customers excluding i
            cust_vals = truck_to_cust[j] + cust_depot
            # set large for i
            cust_vals[i] = np.inf
            min_cust = np.min(cust_vals)
            best = min(best, min_cust)
            best_other_excl.append(best)
        max_other_excl = max(best_other_excl) if best_other_excl else 0.0
        overall_max = max(cur_comp, max_other_excl)
        
        # Prefer smaller max, tie-break by smaller cur_comp
        if overall_max < best_max or (overall_max == best_max and cur_comp < best_current):
            best_max = overall_max
            best_idx = i
            best_current = cur_comp
    
    if best_idx is None:
        return None
    # If only one truck, never wait (since customers exist)
    if n_trucks == 1:
        return best_idx
    # Wait is better only if overall_max_wait < overall_max_best (or tie? not wait)
    if best_max > max_wait:
        return None
    # If tie at depot and wait gives same max, check if current_return is smaller?
    if best_max == max_wait and np.linalg.norm(current_position - depot_position) < 1e-6:
        # At depot, waiting gives 0 return, but serving will increase return; we prefer serving to avoid infinite delay?
        # Actually, if wait gives same max (which includes other trucks), we may still serve to ensure progress.
        # Avoid waiting if there are customers to serve.
        return None
    return best_idx