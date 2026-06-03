import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # compute current makespan (max distance to depot among all trucks)
    dists_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    M0 = np.max(dists_to_depot)
    # also store others' distances excluding current truck?
    # For each customer, compute regret
    best_regret = float('inf')
    best_idx = None
    best_t_current = float('inf')
    for i, cust in enumerate(available_customers):
        t_current = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        M_current = max(t_current, np.max(dists_to_depot))  # includes current truck's current distance? Actually dists_to_depot includes current truck's distance, but we replace with t_current for its truck. However, M_current should consider new distance for current truck and others unchanged. So we need to compute max of t_current and the max of other trucks' distances. So:
        # Compute max of other trucks' distances
        # We can compute overall max, but need to exclude current truck's contribution? Actually dists_to_depot includes current, but if current's distance is the max, we need to replace it with t_current. So M_current = max(t_current, max_{j != current} dists_to_depot[j])
        # So we need index of current truck. We don't have it directly, but we can compute the max of all dists except the current? We can compute by finding the max among all dists and then adjusting if current is the max.
        # Simpler: compute M_others = max(dists_to_depot) but if the index of current truck is the max, we need second max. But we don't have index. We can compute all distances and find the max of all except the current truck's distance. Since truck_positions includes all trucks including current, we need to know which row corresponds to current truck. That information is not provided explicitly. The function signature has current_position as separate, not an index. So we need to find the index of the truck matching current_position. We can do that by comparing positions (with tolerance). That's what the parent code did implicitly? Actually, parent code did not need to differentiate current truck in alternative cost because it computed alt_costs for all trucks including current, then excluded the smallest if it matched cost_now. Here we need the max of other trucks' distances to depot. We can find the index of current truck by np.where(np.all(np.isclose(truck_positions, current_position), axis=1))[0][0]. Then M_others = np.max(np.delete(dists_to_depot, idx)). Alternatively, we can compute without deletion by two largest.
        # Let's find idx
        idx_current = np.where(np.all(np.isclose(truck_positions, current_position), axis=1))[0][0]
        M_others = np.max(np.delete(dists_to_depot, idx_current))
        M_current = max(t_current, M_others)
        # Now compute best alternative makespan: min over other trucks and doing nothing
        best_alt_M = M0  # doing nothing
        for j, truck in enumerate(truck_positions):
            if j == idx_current:
                continue
            t_alt = np.linalg.norm(truck - cust) + np.linalg.norm(cust - depot_position)
            # For alternative truck j, others are all except j
            M_others_alt = np.max(np.delete(dists_to_depot, j))
            M_alt = max(t_alt, M_others_alt)
            if M_alt < best_alt_M:
                best_alt_M = M_alt
        regret = M_current - best_alt_M
        if regret < best_regret or (np.isclose(regret, best_regret) and t_current < best_t_current):
            best_regret = regret
            best_idx = i
            best_t_current = t_current
    # Wait only if at depot and best_regret > 0
    if best_regret > 0 and np.linalg.norm(current_position - depot_position) < 1e-6:
        return None
    return best_idx