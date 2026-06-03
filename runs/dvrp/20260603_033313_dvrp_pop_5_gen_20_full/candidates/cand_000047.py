import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    best_idx = None
    best_cost = float('inf')
    fallback_idx = None
    fallback_cost = float('inf')
    for i, cust in enumerate(available_customers):
        dist_to_cust = np.linalg.norm(current_position - cust)
        dist_cust_to_depot = np.linalg.norm(cust - depot_position)
        cost_now = dist_to_cust + dist_cust_to_depot
        # compute distances from all trucks to this customer
        all_dists = [np.linalg.norm(truck - cust) for truck in truck_positions]
        current_dist = all_dists[0]  # assuming current_position is first truck? Need to find index of current truck
        # Actually we don't know which truck is current. We need to identify which truck in truck_positions is at current_position.
        # Identify index of current truck by position tolerance
        current_idx = None
        for j, pos in enumerate(truck_positions):
            if np.linalg.norm(pos - current_position) < 1e-6:
                current_idx = j
                break
        if current_idx is None:
            # fallback: assume first? Should not happen based on spec
            current_idx = 0
        # Compute min distance among other trucks
        min_other = float('inf')
        for j, d in enumerate(all_dists):
            if j != current_idx:
                if d < min_other:
                    min_other = d
        if current_dist < min_other - 1e-9:  # strictly nearest
            if cost_now < best_cost:
                best_cost = cost_now
                best_idx = i
        # Update fallback
        if cost_now < fallback_cost:
            fallback_cost = cost_now
            fallback_idx = i
    if best_idx is not None:
        return best_idx
    else:
        return fallback_idx