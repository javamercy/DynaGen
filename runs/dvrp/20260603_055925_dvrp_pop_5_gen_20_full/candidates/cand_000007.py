import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    n_trucks = truck_positions.shape[0]
    # Compute distances
    dist_to_cust = np.linalg.norm(current_position - available_customers, axis=1)
    dist_cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    immediate_total = dist_to_cust + dist_cust_to_depot
    
    if n_trucks == 1:
        # Only one truck, must serve, choose the smallest immediate_total
        best_idx = int(np.argmin(immediate_total))
        return best_idx
    
    # Compute average other total for each customer
    other_totals = np.zeros(available_customers.shape[0])
    for i, cust in enumerate(available_customers):
        sum_other = 0.0
        count = 0
        for j, pos in enumerate(truck_positions):
            if np.array_equal(pos, current_position):
                continue
            other_dist = np.linalg.norm(pos - cust) + dist_cust_to_depot[i]
            sum_other += other_dist
            count += 1
        other_totals[i] = sum_other / count
    
    regret = immediate_total - other_totals
    
    # Find customers with negative regret
    negative_mask = regret < 0
    if not np.any(negative_mask):
        return None
    # Among those, pick the most negative regret
    best_idx = int(np.argmin(regret))  # argmin works with negative values
    # If tie, we already got the first min, but we can refine:
    min_regret = regret[best_idx]
    candidates = np.where(regret == min_regret)[0]
    if len(candidates) > 1:
        # Tie-break by smallest immediate_total
        best_idx = int(candidates[np.argmin(immediate_total[candidates])])
    return best_idx