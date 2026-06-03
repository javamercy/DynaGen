import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    # Identify current truck index in truck_positions
    dist_to_current = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(dist_to_current)
    # Precompute customer-depot distances once
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    # This truck's cost to serve each customer
    this_cost = np.linalg.norm(available_customers - current_position, axis=1) + cust_to_depot
    # Best other truck's cost for each customer
    other_costs = np.full(len(available_customers), np.inf)
    for j, pos in enumerate(truck_positions):
        if j == current_idx:
            continue
        other_cost = np.linalg.norm(available_customers - pos, axis=1) + cust_to_depot
        other_costs = np.minimum(other_costs, other_cost)
    # If no other trucks, raw regret is undefined; set to -inf (no advantage)
    if np.isinf(other_costs).all():
        # Only one truck: serve customer with smallest this_cost?
        # But to be safe, maybe always serve? Wait might also be ok.
        # We'll serve the one with smallest this_cost (TSP-like).
        return int(np.argmin(this_cost))
    # Raw regret: positive means this truck better
    raw_regret = other_costs - this_cost
    # If maximum raw regret <= 0, no advantage -> wait
    if np.max(raw_regret) <= 0:
        return None
    # Otherwise, pick customer with max raw regret, tie-break by min this_cost
    max_regret = np.max(raw_regret)
    candidates = np.where(raw_regret == max_regret)[0]
    best_idx = candidates[np.argmin(this_cost[candidates])]
    return int(best_idx)