import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    max_depot_dist = max(dist(pos, depot_position) for pos in truck_positions)
    current_dist = dist(current_position, depot_position)

    ratio = current_dist / max_depot_dist if max_depot_dist > 0 else 1.0
    alpha = 0.2 + 0.3 * ratio  # range 0.2 to 0.5

    best_neg_idx = None
    best_neg_regret = float('inf')
    best_neg_imm = float('inf')
    best_pos_idx = None
    best_pos_regret = float('inf')
    best_pos_imm = float('inf')

    for i, cust in enumerate(available_customers):
        immediate = dist(current_position, cust) + dist(cust, depot_position)

        best_other = float('inf')
        for j, pos in enumerate(truck_positions):
            if np.array_equal(pos, current_position):
                continue
            deferred = dist(pos, cust) + dist(cust, depot_position)
            if deferred < best_other:
                best_other = deferred

        regret = immediate - best_other

        if regret < 0:
            if regret < best_neg_regret or (regret == best_neg_regret and immediate < best_neg_imm):
                best_neg_regret = regret
                best_neg_imm = immediate
                best_neg_idx = i
        else:
            if regret < alpha * current_dist:
                if regret < best_pos_regret or (regret == best_pos_regret and immediate < best_pos_imm):
                    best_pos_regret = regret
                    best_pos_imm = immediate
                    best_pos_idx = i

    if best_neg_idx is not None:
        return best_neg_idx
    return best_pos_idx