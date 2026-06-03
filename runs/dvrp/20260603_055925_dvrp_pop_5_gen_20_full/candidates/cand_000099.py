import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    n_trucks = len(truck_positions)
    current_dist = dist(current_position, depot_position)

    # Distances of other trucks to depot
    other_dists = []
    for pos in truck_positions:
        if not np.array_equal(pos, current_position):
            other_dists.append(dist(pos, depot_position))

    if other_dists:
        max_other = max(other_dists)
        mean_other = np.mean(other_dists)
        std_other = np.std(other_dists)
        imbalance = std_other / (mean_other + 1e-8)
    else:
        max_other = current_dist
        mean_other = current_dist
        imbalance = 0.0

    ratio = current_dist / (max_other + 1e-8)
    ratio = min(ratio, 2.0)  # cap

    base_mult = 0.2
    adaptive_mult = base_mult * (1 + imbalance)
    adaptive_mult = min(adaptive_mult, 1.0)  # cap to avoid too aggressive
    threshold = adaptive_mult * ratio * current_dist

    best_idx = None
    best_regret = float('inf')
    best_imm = float('inf')

    for i, cust in enumerate(available_customers):
        cust_depot = dist(cust, depot_position)
        imm = dist(current_position, cust) + cust_depot

        # Best other truck's total for this customer
        best_other = float('inf')
        for j, pos in enumerate(truck_positions):
            if np.array_equal(pos, current_position):
                continue
            other_val = dist(pos, cust) + cust_depot
            if other_val < best_other:
                best_other = other_val

        regret = imm - best_other if n_trucks > 1 else -1.0

        # Qualification: negative regret always, positive only if below threshold
        qualified = regret < 0 or regret < threshold
        if qualified:
            if regret < best_regret or (regret == best_regret and imm < best_imm):
                best_regret = regret
                best_imm = imm
                best_idx = i

    return best_idx