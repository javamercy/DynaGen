import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    n_trucks = len(truck_positions)
    current_dist = dist(current_position, depot_position)

    # Compute max depot distance among all trucks
    max_depot_dist = max(dist(pos, depot_position) for pos in truck_positions)

    # Average other trucks' depot distance
    other_dists = []
    for pos in truck_positions:
        if not np.array_equal(pos, current_position):
            other_dists.append(dist(pos, depot_position))
    avg_other = np.mean(other_dists) if other_dists else current_dist

    # Base factor from parent: make farther truck more willing
    factor = current_dist / max(avg_other, 1e-6)
    factor = min(factor, 2.0)

    # Fleet-wide multiplier: increase alpha when max_depot_dist is large relative to current_dist
    multiplier = 1.0 + 0.5 * (max_depot_dist / (current_dist + 1e-6))
    multiplier = min(multiplier, 3.0)

    alpha = 0.2 * factor * multiplier

    best_neg = None
    best_neg_regret = float('inf')
    best_neg_immediate = float('inf')
    best_pos = None
    best_pos_regret = float('inf')
    best_pos_immediate = float('inf')

    for i, cust in enumerate(available_customers):
        cust_return = dist(cust, depot_position)
        immediate_total = dist(current_position, cust) + cust_return

        if n_trucks > 1:
            best_other = float('inf')
            for j, pos in enumerate(truck_positions):
                if np.array_equal(pos, current_position):
                    continue
                deferred = dist(pos, cust) + cust_return
                if deferred < best_other:
                    best_other = deferred
            regret = immediate_total - best_other
        else:
            regret = -1.0

        if regret < 0:
            if regret < best_neg_regret or (regret == best_neg_regret and immediate_total < best_neg_immediate):
                best_neg_regret = regret
                best_neg_immediate = immediate_total
                best_neg = i
        else:
            if regret < alpha * current_dist:
                if regret < best_pos_regret or (regret == best_pos_regret and immediate_total < best_pos_immediate):
                    best_pos_regret = regret
                    best_pos_immediate = immediate_total
                    best_pos = i

    if best_neg is not None:
        return best_neg
    return best_pos