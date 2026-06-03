import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    n_trucks = len(truck_positions)
    best_candidate = None
    best_regret = float('inf')
    best_dist = float('inf')

    # First pass: negative regret customers
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
            if regret < best_regret or (regret == best_regret and immediate_total < best_dist):
                best_regret = regret
                best_dist = immediate_total
                best_candidate = i

    if best_candidate is not None:
        return best_candidate

    # Second pass: customers with small positive overhead
    best_overhead = float('inf')
    best_immediate = float('inf')
    best_candidate = None
    threshold = 0.1  # relative overhead threshold

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
            if best_other <= 0:
                continue
            overhead = (immediate_total - best_other) / best_other
        else:
            overhead = -1.0  # single truck: always serve (negligible overhead)

        if overhead <= threshold:
            if overhead < best_overhead or (overhead == best_overhead and immediate_total < best_immediate):
                best_overhead = overhead
                best_immediate = immediate_total
                best_candidate = i

    return best_candidate