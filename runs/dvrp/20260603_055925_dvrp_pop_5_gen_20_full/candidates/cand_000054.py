import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    n_trucks = len(truck_positions)
    n_available = len(available_customers)

    best_candidate = None
    best_regret = float('inf')
    best_immediate_dist = float('inf')

    # First pass: compute regrets and identify negative regret candidates
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
            if regret < best_regret or (regret == best_regret and immediate_total < best_immediate_dist):
                best_regret = regret
                best_immediate_dist = immediate_total
                best_candidate = i

    if best_candidate is not None:
        return best_candidate

    # No negative regret: apply adaptive threshold
    dist_to_depot = dist(current_position, depot_position)

    # Adaptive alpha based on customer density per truck
    density = n_available / max(n_trucks, 1)
    alpha = 0.1 + 0.4 * (density / 5.0)  # tune: baseline 0.1, max 0.5 when density >=5
    alpha = min(max(alpha, 0.1), 0.5)

    wait_threshold = alpha * dist_to_depot
    best_candidate = None
    best_regret = float('inf')
    best_immediate_dist = float('inf')

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

        if regret < wait_threshold and (regret < best_regret or (regret == best_regret and immediate_total < best_immediate_dist)):
            best_regret = regret
            best_immediate_dist = immediate_total
            best_candidate = i

    return best_candidate