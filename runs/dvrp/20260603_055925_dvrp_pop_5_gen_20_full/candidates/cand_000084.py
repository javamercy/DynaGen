import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    n_trucks = len(truck_positions)
    current_dist = dist(current_position, depot_position)

    # Compute max depot distance of other trucks
    other_dists = [dist(pos, depot_position) for pos in truck_positions if not np.array_equal(pos, current_position)]
    max_other = max(other_dists) if other_dists else current_dist

    # Adaptive threshold: ratio of current distance to max of others, capped
    ratio = current_dist / (max_other + 1e-8)
    ratio = min(ratio, 3.0)  # increased cap
    threshold = 0.35 * ratio * current_dist  # increased multiplier

    best_idx = None
    best_regret = float('inf')
    best_imm = float('inf')  # now used for current-customer distance only

    for i, cust in enumerate(available_customers):
        cust_depot = dist(cust, depot_position)
        imm = dist(current_position, cust)  # only current-customer distance, not including depot

        # Compute best total for other trucks
        best_other = float('inf')
        for j, pos in enumerate(truck_positions):
            if np.array_equal(pos, current_position):
                continue
            other_val = dist(pos, cust) + cust_depot
            if other_val < best_other:
                best_other = other_val

        regret = (dist(current_position, cust) + cust_depot) - best_other if n_trucks > 1 else -1.0

        # Qualification: negative regret always, positive only if below threshold
        qualified = regret < 0 or regret < threshold
        if qualified:
            if regret < best_regret or (regret == best_regret and imm < best_imm):
                best_regret = regret
                best_imm = imm
                best_idx = i

    return best_idx