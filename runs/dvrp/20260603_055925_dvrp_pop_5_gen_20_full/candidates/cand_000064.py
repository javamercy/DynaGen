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
            max_deferred = -float('inf')
            for j, pos in enumerate(truck_positions):
                if np.array_equal(pos, current_position):
                    continue
                deferred = dist(pos, cust) + cust_return
                if deferred > max_deferred:
                    max_deferred = deferred
            regret = immediate_total - max_deferred
        else:
            regret = -1.0

        if regret < 0:
            if regret < best_regret or (regret == best_regret and immediate_total < best_immediate):
                best_regret = regret
                best_immediate = immediate_total
                best_candidate = i

    return best_candidate