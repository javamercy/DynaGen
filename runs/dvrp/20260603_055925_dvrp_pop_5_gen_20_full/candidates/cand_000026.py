import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    n = len(truck_positions)
    # direct return times for each truck to depot
    direct = np.array([dist(pos, depot_position) for pos in truck_positions])
    baseline_max = np.max(direct)

    # identify index of current truck in truck_positions
    # assume current_position exactly matches one row
    current_idx = np.argmin(np.sum((truck_positions - current_position) ** 2, axis=1))

    # compute max of direct returns for other trucks
    others_direct = np.delete(direct, current_idx)
    max_others = np.max(others_direct) if len(others_direct) > 0 else -np.inf

    best_idx = None
    best_new_max = float('inf')
    best_this_return = float('inf')

    for i, cust in enumerate(available_customers):
        this_return = dist(current_position, cust) + dist(cust, depot_position)
        new_max = max(this_return, max_others)
        if new_max < baseline_max:
            if new_max < best_new_max or (new_max == best_new_max and this_return < best_this_return):
                best_new_max = new_max
                best_this_return = this_return
                best_idx = i

    return best_idx