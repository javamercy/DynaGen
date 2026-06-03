import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    n_trucks = len(truck_positions)
    current_dist = dist(current_position, depot_position)

    # Average depot distance of other trucks
    other_dists = [dist(pos, depot_position) for pos in truck_positions if not np.array_equal(pos, current_position)]
    avg_other = np.mean(other_dists) if other_dists else current_dist

    # Imbalance-responsive threshold
    # alpha increases when current truck is farther than average
    relative_gap = max(0.0, (current_dist - avg_other) / (avg_other + 1e-8))
    alpha = 0.2 + 0.3 * min(relative_gap, 1.0)  # cap at 0.5
    ratio = current_dist / (avg_other + 1e-8)
    ratio = min(ratio, 2.0)
    threshold = alpha * ratio * current_dist

    best_idx = None
    best_regret = float('inf')
    best_imm = float('inf')

    for i, cust in enumerate(available_customers):
        cust_depot = dist(cust, depot_position)
        imm = dist(current_position, cust) + cust_depot

        # Best total for other trucks (excluding current)
        best_other = float('inf')
        for pos in truck_positions:
            if np.array_equal(pos, current_position):
                continue
            other_val = dist(pos, cust) + cust_depot
            if other_val < best_other:
                best_other = other_val

        regret = imm - best_other if n_trucks > 1 else -1.0

        qualified = regret < 0 or regret < threshold
        if qualified:
            if regret < best_regret or (regret == best_regret and imm < best_imm):
                best_regret = regret
                best_imm = imm
                best_idx = i

    return best_idx