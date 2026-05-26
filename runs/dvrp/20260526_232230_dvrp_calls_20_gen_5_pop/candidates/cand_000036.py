import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b, axis=-1)

    # current truck's direct return distance to depot
    current_return = np.linalg.norm(current_position - depot_position)

    # other trucks' return distances to depot
    other_returns = np.linalg.norm(truck_positions - depot_position, axis=-1)
    # exclude current truck? we don't know which index; but we assume truck_positions includes all trucks including current?
    # To be safe, compute max of other trucks only. Since we have current_return separately.
    # If truck_positions includes the current truck, we need to exclude it. But we don't have index.
    # Simpler: compute max over all trucks, then later compare with current_return.
    # Actually we want max of other trucks, but we can compute max_over_all = max of all trucks.
    # But current_return is already one of them. So max_over_all = max(current_return, others_max). So we can compute all returns and then subtract current?
    # Let's compute all returns including current, then set other_max = max over all returns except current? But we don't know index.
    # Alternative: assume truck_positions may contain current_position as one row. Compute diff all vs depot: all_returns = np.linalg.norm(truck_positions - depot_position, axis=-1).
    # Then other_max = max of all_returns where truck != current? But no index. 
    # Instead, compute other_max as np.max(all_returns) but then current_return is included. If we want other_max, we can compute current_return separately.
    # Since current_return is just one value, the max over all trucks is max(current_return, others_max). So we can compute all_returns_max = np.max(np.linalg.norm(truck_positions - depot_position, axis=-1)).
    # Then if all_returns_max > current_return, then others_max = all_returns_max; else others_max = current_return? Not exactly.
    # Better: we know current_position is one of the trucks. We can compute all returns and then remove current_return by creating a mask.
    # But we don't have index. We could compute distances between each truck and current_position to identify which row is current.
    # That is too heavy. Let's assume truck_positions does NOT include current truck? Actually the function signature: truck_positions (n_trucks, 2) all trucks. So includes current truck.
    # We'll compute all returns and then find the index where distance to current_position is 0.
    diffs = truck_positions - current_position
    # find index where all zeros
    idx_current = np.where(np.all(diffs == 0, axis=1))[0]
    if len(idx_current) == 0:
        # current not found? treat as separate.
        other_returns = np.linalg.norm(truck_positions - depot_position, axis=-1)
    else:
        mask = np.ones(truck_positions.shape[0], dtype=bool)
        mask[idx_current[0]] = False
        other_returns = np.linalg.norm(truck_positions[mask] - depot_position, axis=-1)
    
    if len(other_returns) == 0:
        other_max = 0.0
    else:
        other_max = np.max(other_returns)

    current_max = max(current_return, other_max)

    best_idx = None
    best_new_max = float('inf')
    best_new_return = float('inf')

    for i, customer in enumerate(available_customers):
        d_to_cust = np.linalg.norm(current_position - customer)
        d_cust_to_depot = np.linalg.norm(customer - depot_position)
        new_return = d_to_cust + d_cust_to_depot
        new_max = max(new_return, other_max)

        # If going to this customer would increase the max beyond current_max, skip?
        # But we only want to consider customers that do not worsen the max?
        # According to reflection, we return None if ALL customers worsen. So we need to track if any improves.
        if new_max < current_max:
            # improvement
            if (new_max < best_new_max or
                (new_max == best_new_max and new_return < best_new_return)):
                best_new_max = new_max
                best_new_return = new_return
                best_idx = i
        elif new_max == current_max:
            # tie: consider as acceptable
            if best_idx is None or new_max < best_new_max or (new_max == best_new_max and new_return < best_new_return):
                best_new_max = new_max
                best_new_return = new_return
                best_idx = i
        # else new_max > current_max: skip

    # If no customer kept current_max or improved, return None
    if best_idx is None:
        return None
    else:
        return best_idx