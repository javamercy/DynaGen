import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if len(available_customers) == 0:
        return None
    # Current distance from each truck to depot
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # Current max return time (assuming all trucks go directly back)
    current_max_return = np.max(dist_to_depot)
    # Active truck index: the one at current_position (should match one of truck_positions)
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    # Precompute distances from each available customer to each other truck (excluding active)
    # To avoid recomputing in loop, we can compute distances to all trucks and then exclude active.
    # But not necessary; loop is fine for small sets.
    best_customer = None
    best_score = np.inf
    best_active_return = np.inf
    # Weights
    alpha = 0.1   # gap penalty
    beta = 0.05   # spatial penalty
    for i, cust in enumerate(available_customers):
        # Distance from active truck to customer
        d_active_to_cust = np.linalg.norm(current_position - cust)
        # Estimated return time for active truck if serves customer
        active_return = d_active_to_cust + np.linalg.norm(cust - depot_position)
        # New return times array: active truck's updated, others remain current
        new_return_times = dist_to_depot.copy()
        new_return_times[active_idx] = active_return
        new_max_return = np.max(new_return_times)
        # Gap: how much active's return exceeds current max (before serving)
        # Actually we want to penalize if active's return is far below max (i.e., active is underutilized)?
        # Reflection says penalize the difference between active truck return and current max return to encourage catching up.
        # So gap = current_max_return - active_return (positive if active is behind). Larger gap means active needs to speed up.
        gap = current_max_return - active_return
        # Spatial penalty: distance to nearest other truck (excluding active)
        other_truck_dists = np.linalg.norm(truck_positions - cust, axis=1)
        # Set distance to active truck to inf so it's not considered
        other_truck_dists[active_idx] = np.inf
        min_dist_other = np.min(other_truck_dists)
        # Score: maximize new_max_return? No, minimize new_max_return, but also minimize gap penalty (since gap penalizes when active is behind? Actually we want to reduce gap, so larger gap should be penalized positively? Wait: if gap is large, active is far behind current max, we want to serve customer that reduces that gap? Serving a customer reduces the gap if it increases active's return? Actually gap = current_max - active_return; if we serve a customer, active_return increases, so gap decreases. So we want to serve customers that reduce gap (make active catch up). So a positive gap is good (reducing it), so we want to penalize small gap? Let's interpret: The reflection says "penalizing the gap between the active truck's return and the fleet maximum". Typically, if the gap is large, the active truck is behind and we want to encourage it to catch up. So we want to reward serving customers that bring active's return closer to max, i.e., reduce gap. So the penalty should be proportional to the gap? Actually if we include gap as a term in the score, we want to minimize score, so a smaller score should be better. If gap is large (active behind), serving that customer increases active_return and reduces gap, so we might want to prioritize customers that result in a smaller gap (i.e., bigger reduction). But the gap is computed after serving? The description is ambiguous. I'll interpret as: we want to penalize the active truck if its return is far below the max, so we add a term that is larger when active_return is much smaller than current_max. That term will encourage choosing customers that increase active_return, thus reducing the gap. So I'll use gap = current_max_return - active_return (active_return after serving), which is positive if active is behind. This gap is the amount by which active's return is less than current max. To penalize being behind, we add alpha * gap to score. Then a larger gap (worse) makes score larger, so we avoid those customers? Wait: if active is behind, serving a customer increases active_return, reducing gap. So the gap after serving is smaller than current gap. So we want to prefer customers that result in smaller gap. So adding alpha * gap to score means we prefer smaller gap (good). So this is okay: smaller gap gives lower score. So we want to minimize gap. But then we already minimize new_max_return, which indirectly reduces gap. Adding gap explicitly may help.
        # Spatial: smaller min_dist_other is bad (trucks clustered), so we want larger min_dist_other -> lower score? So we subtract beta*min_dist_other or add positively? We want to encourage larger min_dist, so we should add a penalty that is larger when min_dist is small. So use penalty = -beta * min_dist_other (negative) or add beta * (some max distance - min_dist). Simpler: score = new_max_return + alpha * gap - beta * min_dist_other. But we need to ensure consistency: lower score better. If min_dist is large (good spread), subtracting it lowers score, encouraging that customer. That seems correct.
        score = new_max_return + alpha * gap - beta * min_dist_other
        # Tie-break by active_return (prefer smaller active return? Actually tie-breaking by active_return as in parent: if scores equal, choose customer with smaller active_return. That might favor shorter trips? But okay.
        if score < best_score or (score == best_score and active_return < best_active_return):
            best_score = score
            best_active_return = active_return
            best_customer = i
    return best_customer