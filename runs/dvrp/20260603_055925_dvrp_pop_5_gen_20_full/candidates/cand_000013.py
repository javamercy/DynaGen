import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    n_trucks = len(truck_positions)
    best_candidate = None
    best_regret = float('inf')
    best_immediate = float('inf')

    for i, cust in enumerate(available_customers):
        cust_return = dist(cust, depot_position)
        immediate_total = dist(current_position, cust) + cust_return

        if n_trucks > 1:
            deferred_sum = 0.0
            count = 0
            for j, pos in enumerate(truck_positions):
                if np.array_equal(pos, current_position):
                    continue
                deferred = dist(pos, cust) + cust_return
                deferred_sum += deferred
                count += 1
            if count > 0:
                avg_deferred = deferred_sum / count
                regret = immediate_total - avg_deferred
            else:
                regret = -1.0
        else:
            regret = -1.0

        if regret < 0:
            if regret < best_regret or (regret == best_regret and immediate_total < best_immediate):
                best_regret = regret
                best_immediate = immediate_total
                best_candidate = i

    return best_candidate