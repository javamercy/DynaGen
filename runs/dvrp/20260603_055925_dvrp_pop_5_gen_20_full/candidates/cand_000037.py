import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    n_trucks = len(truck_positions)
    # Special case: single truck
    if n_trucks == 1:
        best_idx = None
        best_total = float('inf')
        for i, cust in enumerate(available_customers):
            total = dist(current_position, cust) + dist(cust, depot_position)
            if total < best_total:
                best_total = total
                best_idx = i
        return best_idx

    # Baseline max return time if all trucks return directly
    baseline_max = max(dist(pos, depot_position) for pos in truck_positions)

    best_idx = None
    best_new_max = float('inf')
    best_immediate = float('inf')

    for i, cust in enumerate(available_customers):
        immediate_total = dist(current_position, cust) + dist(cust, depot_position)
        # Max return time among other trucks if they return directly
        max_other = -1.0
        for j, pos in enumerate(truck_positions):
            if np.array_equal(pos, current_position):
                continue
            d = dist(pos, depot_position)
            if d > max_other:
                max_other = d
        new_max = max(immediate_total, max_other)
        if new_max < best_new_max or (new_max == best_new_max and immediate_total < best_immediate):
            best_new_max = new_max
            best_immediate = immediate_total
            best_idx = i

    if best_new_max > baseline_max:
        return None
    else:
        return best_idx