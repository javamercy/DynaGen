import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    current_dist = dist(current_position, depot_position)

    best_idx = None
    best_regret = float('inf')
    best_imm = float('inf')

    for i, cust in enumerate(available_customers):
        current_to_cust = dist(current_position, cust)
        cust_depot = dist(cust, depot_position)
        imm = current_to_cust + cust_depot

        min_other = float('inf')
        for pos in truck_positions:
            if np.array_equal(pos, current_position):
                continue
            other_to_cust = dist(pos, cust)
            if other_to_cust < min_other:
                min_other = other_to_cust

        if len(truck_positions) == 1:
            regret = -float('inf')
        else:
            regret = current_to_cust - min_other

        threshold = 0.2 * current_dist if current_dist > 0 else 0.0
        if regret < 0 or regret < threshold:
            if regret < best_regret or (regret == best_regret and imm < best_imm):
                best_regret = regret
                best_imm = imm
                best_idx = i

    return best_idx