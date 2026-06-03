import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    n_trucks = len(truck_positions)
    current_to_depot = dist(current_position, depot_position)
    bias = 0.1 * current_to_depot

    best_candidate = None
    best_regret = float('inf')
    best_immediate = float('inf')

    for i, cust in enumerate(available_customers):
        cust_to_depot = dist(cust, depot_position)
        immediate_total = dist(current_position, cust) + cust_to_depot

        if n_trucks > 1:
            best_other = float('inf')
            for j, pos in enumerate(truck_positions):
                if np.array_equal(pos, current_position):
                    continue
                deferred = dist(pos, cust) + cust_to_depot
                if deferred < best_other:
                    best_other = deferred
            biased_regret = immediate_total - (best_other + bias)
        else:
            biased_regret = -1.0  # always negative to force serve

        if biased_regret < 0:
            if biased_regret < best_regret or (biased_regret == best_regret and immediate_total < best_immediate):
                best_regret = biased_regret
                best_immediate = immediate_total
                best_candidate = i

    if best_candidate is None:
        # Fallback: minimal immediate total (greedy)
        min_immediate = float('inf')
        best_candidate = None
        for i, cust in enumerate(available_customers):
            immediate_total = dist(current_position, cust) + dist(cust, depot_position)
            if immediate_total < min_immediate:
                min_immediate = immediate_total
                best_candidate = i
        return best_candidate
    else:
        return best_candidate