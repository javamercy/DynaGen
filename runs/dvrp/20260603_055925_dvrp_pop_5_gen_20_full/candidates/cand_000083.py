import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    n_trucks = len(truck_positions)
    
    # Compute threshold as 0.2 * mean distance of all trucks to depot
    all_dists = [dist(pos, depot_position) for pos in truck_positions]
    mean_dist = np.mean(all_dists)
    threshold = 0.2 * mean_dist

    best_idx = None
    best_regret = float('inf')
    best_imm = float('inf')

    for i, cust in enumerate(available_customers):
        cust_depot = dist(cust, depot_position)
        imm = dist(current_position, cust) + cust_depot

        # Compute best total for other trucks
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