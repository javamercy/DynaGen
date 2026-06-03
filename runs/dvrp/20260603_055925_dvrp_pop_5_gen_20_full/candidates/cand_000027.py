import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    n_trucks = len(truck_positions)
    if n_trucks == 1:
        distances = np.linalg.norm(available_customers - current_position, axis=1)
        return int(np.argmin(distances))
    # Find current truck index
    current_truck_idx = int(np.argmin(np.linalg.norm(truck_positions - current_position, axis=1)))
    # Current distances to depot for all trucks
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max_depot = np.max(dist_to_depot)

    best_score = -float('inf')
    best_idx = None
    best_regret = -float('inf')
    distance_tiebreaker = float('inf')

    for i, cust in enumerate(available_customers):
        d_curr = np.linalg.norm(current_position - cust)
        cust_depot = np.linalg.norm(cust - depot_position)
        current_cost = d_curr + cust_depot
        # Compute best other cost
        other_costs = []
        for j, pos in enumerate(truck_positions):
            if j == current_truck_idx:
                continue
            d_other = np.linalg.norm(pos - cust)
            other_cost = d_other + cust_depot
            other_costs.append(other_cost)
        best_other = min(other_costs) if other_costs else float('inf')
        regret = best_other - current_cost
        if regret < 0:
            continue  # negative regret, skip
        # Compute penalty based on max distance to depot
        new_current_return = current_cost  # because current_cost = d_curr + cust_depot
        new_max_depot = max(new_current_return, np.max(np.delete(dist_to_depot, current_truck_idx)))
        penalty = 0.1 * (new_max_depot - current_max_depot)
        score = regret - penalty
        # Tie-breaking: if scores equal, prefer smaller d_curr
        if score > best_score or (score == best_score and d_curr < distance_tiebreaker):
            best_score = score
            best_idx = i
            best_regret = regret
            distance_tiebreaker = d_curr

    if best_idx is not None and best_regret >= 0:
        return best_idx
    else:
        return None