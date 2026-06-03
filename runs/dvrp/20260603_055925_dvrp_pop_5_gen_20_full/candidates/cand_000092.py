import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    n_trucks = len(truck_positions)
    n_available = len(available_customers)

    distances_to_depot = [dist(pos, depot_position) for pos in truck_positions]
    current_dist_to_depot = dist(current_position, depot_position)
    max_fleet_dist_to_depot = max(distances_to_depot)

    # Precompute all distances from each truck to each customer
    # For efficiency
    cust_to_depot = np.array([dist(c, depot_position) for c in available_customers])
    dist_current_to_cust = np.array([dist(current_position, c) for c in available_customers])
    immediate_total = dist_current_to_cust + cust_to_depot

    # Compute regret for each customer
    regrets = np.zeros(n_available)
    for i in range(n_available):
        if n_trucks > 1:
            best_other = float('inf')
            for j, pos in enumerate(truck_positions):
                if np.array_equal(pos, current_position):
                    continue
                deferred = dist(pos, available_customers[i]) + cust_to_depot[i]
                if deferred < best_other:
                    best_other = deferred
            regrets[i] = immediate_total[i] - best_other
        else:
            regrets[i] = -1.0

    # Check for negative regret
    neg_mask = regrets < 0
    if np.any(neg_mask):
        # Among negative, smallest regret (most negative) then smallest immediate_total
        neg_indices = np.where(neg_mask)[0]
        best_idx = neg_indices[np.argmin(regrets[neg_mask])]
        # Tie-break by immediate_total if multiple have same regret (rare)
        best_regret = regrets[best_idx]
        same_regret = neg_indices[regrets[neg_indices] == best_regret]
        if len(same_regret) > 1:
            best_idx = same_regret[np.argmin(immediate_total[same_regret])]
        return int(best_idx)

    # No negative regret: adaptive threshold
    density = n_available / max(n_trucks, 1)
    alpha = 0.1 + 0.4 * min(density / 5.0, 1.0)  # 0.1 to 0.5
    # Increase alpha if fleet is imbalanced
    if max_fleet_dist_to_depot > 0 and current_dist_to_depot > 0:
        imbalance = max_fleet_dist_to_depot / current_dist_to_depot
        if imbalance > 2.0:
            alpha = min(alpha + 0.2, 0.7)

    # Weight max distance more
    threshold = alpha * (current_dist_to_depot + 2 * max_fleet_dist_to_depot) / 3.0

    # Find best customer with regret < threshold
    eligible = np.where(regrets < threshold)[0]
    if len(eligible) == 0:
        return None
    # Among eligible, smallest regret then smallest immediate_total
    best_idx = eligible[np.argmin(regrets[eligible])]
    best_regret = regrets[best_idx]
    same = eligible[regrets[eligible] == best_regret]
    if len(same) > 1:
        best_idx = same[np.argmin(immediate_total[same])]
    return int(best_idx)