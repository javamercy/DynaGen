import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None

    # Compute distances from each truck to depot
    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    
    # For active truck, its current distance to depot
    active_dist = np.linalg.norm(current_position - depot_position)
    
    # For each available customer, compute new return time for active truck
    # dist(current, customer) + dist(customer, depot)
    dist_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    new_return_times = dist_to_cust + cust_to_depot
    
    # Current max of other trucks' distances (excluding active truck? Actually we don't know index of active truck in truck_positions. We'll compute max of all trucks, but active truck's distance will be replaced by new_return_time, so we need max of others. We can compute max of all trucks, then for each candidate, new_max = max(new_return_time, max_other). But max_other = max of truck_to_depot excluding active? Since we don't know which index is active, we can compute overall max, but careful: If active truck is the current max, then new_max might be different. Better: compute max of all trucks, but then for each candidate, the new max is max(new_return_time, max_of_all_except_active). To get max_of_all_except_active, we can compute overall max and then if active truck is the max, we need the second max. But that's tricky. Instead, we can compute the max of all truck distances, then for each candidate, compute the new max as max(new_return_time, overall_max) but that's an overestimate if active truck is not the max and new_return_time is less than max. Actually, if active truck is not the max, the max stays the same if new_return_time <= max_other, else becomes new_return_time. But we don't know active truck index. Simpler: assume we can identify active truck's distance among truck_positions? The function doesn't provide its index. We can compute distance from current_position to depot, and compare to each truck distance to find which truck is at current_position? But multiple trucks could be at same position? Not likely but possible. We'll just treat all trucks equally and compute overall max including active. Then for each customer, new overall max = max(new_return_time, overall_max) but if active truck is the max, overall_max includes active's current distance, and new_return_time >= that, so it's fine. But if active is not the max, overall_max is from another truck, and new_return_time might be smaller than that max, so new overall max = overall_max, which is correct. So we can just use overall_max (including active). That works because if active is not the max, overall_max stays same even if new_return_time < that max. So it's correct.
    overall_max = np.max(truck_to_depot)
    
    candidate_max = np.maximum(new_return_times, overall_max)
    
    # Choose customer with smallest candidate_max
    best_idx = np.argmin(candidate_max)
    
    # But if multiple with same candidate_max, tie-break by new_return_time (smaller better)
    # Or by distance to customer? We'll use new_return_time.
    # To implement tie-breaking, get indices where candidate_max == candidate_max[best_idx], then pick smallest new_return_time.
    min_val = candidate_max[best_idx]
    ties = np.where(candidate_max == min_val)[0]
    if len(ties) > 1:
        # among ties, pick smallest new_return_time
        best_idx_in_ties = np.argmin(new_return_times[ties])
        best_idx = ties[best_idx_in_ties]
    
    return int(best_idx)