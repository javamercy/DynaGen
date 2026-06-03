import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    n_trucks = len(truck_positions)
    current_dist = dist(current_position, depot_position)
    
    # current makespan
    all_dist_to_depot = [dist(pos, depot_position) for pos in truck_positions]
    current_makespan = max(all_dist_to_depot)
    
    # average depot distance of other trucks
    other_dists = [d for i, d in enumerate(all_dist_to_depot) if not np.array_equal(truck_positions[i], current_position)]
    avg_other = np.mean(other_dists) if other_dists else current_dist
    
    # threshold (same as parent)
    relative_gap = max(0.0, (current_dist - avg_other) / (avg_other + 1e-8))
    alpha = 0.2 + 0.3 * min(relative_gap, 1.0)
    ratio = current_dist / (avg_other + 1e-8)
    ratio = min(ratio, 2.0)
    threshold = alpha * ratio * current_dist
    
    best_idx = None
    best_regret = float('inf')
    best_imm = float('inf')
    
    for i, cust in enumerate(available_customers):
        cust_depot = dist(cust, depot_position)
        imm = dist(current_position, cust) + cust_depot
        
        if n_trucks == 1:
            regret = -1.0
        else:
            # best other total distance
            best_other = float('inf')
            for pos in truck_positions:
                if np.array_equal(pos, current_position):
                    continue
                other_val = dist(pos, cust) + cust_depot
                if other_val < best_other:
                    best_other = other_val
            # makespan impact
            increase_self = max(0.0, imm - current_makespan)
            increase_other = max(0.0, best_other - current_makespan)
            regret = increase_self - increase_other
        
        qualified = regret < 0 or regret < threshold
        if qualified:
            if regret < best_regret or (regret == best_regret and imm < best_imm):
                best_regret = regret
                best_imm = imm
                best_idx = i
    
    return best_idx