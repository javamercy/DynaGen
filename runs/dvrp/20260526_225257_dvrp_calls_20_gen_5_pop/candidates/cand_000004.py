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
    
    # compute return time if this truck returns directly to depot now
    direct_return_time = current_time + np.linalg.norm(current_position - depot_position)
    
    # compute the max return time among other trucks if they return directly
    other_return_times = []
    for i, pos in enumerate(truck_positions):
        # skip the active truck? We have no identifier, but we can assume the active truck is the one closest to current_position? Actually we are given current_position, but multiple trucks could be at same position. However, we assume the active truck is the one making the decision. Since we don't have its index, we exclude any truck whose position equals current_position? But that's fragile. Instead, compute all trucks' direct return times, then we will handle active truck separately.
        other_return_times.append(np.linalg.norm(pos - depot_position))
    # We don't know which truck is active, so we compute max of all, but then for candidate we replace active truck's return time with its new time.
    # We need to identify which index corresponds to active truck. Since we only have positions, we consider the active truck as the one that decides. But there might be duplicates. A safe approach: compute the new max without knowing index by considering that the active truck's return time changes. But if we cannot identify it, we can assume the active truck is the one with same position as current_position? That might be okay but could match multiple. 
    # Better: we know the active truck is the one that will drive. Its current time is current_time. So its direct return time is direct_return_time. Among all trucks, we can subtract that from max? Actually we need to compute max of other trucks' return times (excluding active). So we need to find which truck(s) have the same position? Not guaranteed.
    # Instead, we can compute the max over all trucks' current return times (including active) and then for each candidate compute new max assuming active trucks' time becomes new_time. That works without identifying the active truck: the current max is max_all = max(direct_return_time, max of others). But if we don't identify, we might underestimate others? Actually if multiple trucks at same position, we treat them equally. But then new max after serving will replace the direct_return_time of that specific truck. However, if other trucks also have same position, they are separate. So we need to know how many trucks at that position to properly adjust the max. This is messy.
    # Simpler: assume the active truck is the one that is calling the function. We don't have its index, but we can compute the direct return time for all trucks. Then we can compute the current max excluding the active truck by using the fact that the active truck's direct return time is direct_return_time. So max_other = max( np.linalg.norm(truck_positions[i] - depot_position) for i where not (truck_positions[i] equals current_position)? But equal may not hold due to floating point.
    # Given constraints, we approximate: assume the active truck is the one whose position equals current_position exactly (since it's the same array). We can compare by element. 
    # Use np.allclose maybe.
    # For simplicity in a stateless function, we can use the fact that the active truck's position is passed separately, so we can filter by checking equality of both coordinates. That's safe because positions are double arrays.
    
    # Identify which truck indices are 'active' coordinates identical to current_position
    active_mask = np.all(truck_positions == current_position, axis=1)
    # There could be multiple? In practice exactly one, but handle generic.
    # Compute direct return times for all trucks
    all_direct_times = np.linalg.norm(truck_positions - depot_position, axis=1) + current_time  # Note: current_time is same for all? Actually each truck may have different current_time? But the problem says current_time is the deciding truck's time. Other trucks might have different current times? The snapshot gives only positions; no mention of times for other trucks. So we assume all trucks share the same current_time? That's unrealistic but necessary. Alternatively, we can ignore time differences and only consider distances. Since we are minimizing TTT, the time component matters. But without other trucks' times, we assume they are all at the same current_time. This is a limitation.
    # To be safe, we ignore current_time for other trucks and only use distance. Then the max is max over distances. But that neglects that some trucks may have already delivered and returned partially? Actually current_time is the same for all? In online DVRP, each truck has its own clock. But the function signature passes only one current_time. So we assume all trucks share that time. 
    
    # So compute current max return time among all trucks if they all return now:
    # For active trucks, direct_return_time is computed.
    # For others, direct_return_time_other = current_time + distance to depot.
    # But since current_time is same, we can just use distances.
    other_direct_distances = np.linalg.norm(truck_positions - depot_position, axis=1)
    # For active trucks, we will replace with new time.
    
    best_idx = None
    best_max = np.inf
    
    # Current max if active truck returns now:
    # Replace active truck's distance with direct_return_time (but direct_return_time already uses current_time). 
    # Actually, for computing max, we need to consider the actual return times. Current max return time is:
    # max( direct_return_time, np.max(other_direct_distances + current_time) )
    # But we can simplify by considering that all trucks have same current_time, so max return time = current_time + max distance among all trucks. That is current_max_return = current_time + np.max(np.linalg.norm(truck_positions - depot_position, axis=1)). But if active truck returns now, its return time = direct_return_time. So current_max_return = max( direct_return_time, current_time + max(other distances) ).
    
    current_max = max(direct_return_time, current_time + np.max(other_direct_distances[~active_mask])) if np.any(~active_mask) else direct_return_time
    
    for i, cust in enumerate(available_customers):
        new_active_time = current_time + np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # new max considering active truck's new time and others unchanged
        new_max = max(new_active_time, current_time + np.max(other_direct_distances[~active_mask])) if np.any(~active_mask) else new_active_time
        if new_max < best_max:
            best_max = new_max
            best_idx = i
    
    # If best_max >= current_max, then returning now is at least as good
    if best_max >= current_max:
        return None
    else:
        return best_idx