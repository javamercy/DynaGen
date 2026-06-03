import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    n_trucks = len(truck_positions)
    n_available = len(available_customers)

    # Precompute distances
    dist_to_depot = dist(current_position, depot_position)
    other_dist_to_depot = []
    for pos in truck_positions:
        if not np.array_equal(pos, current_position):
            other_dist_to_depot.append(dist(pos, depot_position))
    max_other_dist_to_depot = max(other_dist_to_depot) if other_dist_to_depot else 0.0
    urgency = max(0.0, dist_to_depot - max_other_dist_to_depot)

    beta = 0.3  # urgency weight

    # Compute effective regret for each customer
    best_candidate = None
    best_effective_regret = float('inf')
    best_immediate_total = float('inf')

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
            base_regret = immediate_total - best_other
        else:
            base_regret = -1.0

        effective_regret = base_regret + beta * urgency

        if effective_regret < 0:
            if (effective_regret < best_effective_regret or
                (effective_regret == best_effective_regret and immediate_total < best_immediate_total)):
                best_effective_regret = effective_regret
                best_immediate_total = immediate_total
                best_candidate = i

    if best_candidate is not None:
        return best_candidate

    # No negative effective regret: adaptive waiting threshold
    # Density-based alpha
    density = n_available / max(n_trucks, 1)
    alpha = 0.1 + 0.4 * (density / 5.0)
    alpha = min(max(alpha, 0.1), 0.5)

    wait_threshold = alpha * dist_to_depot

    best_candidate = None
    best_effective_regret = float('inf')
    best_immediate_total = float('inf')

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
            base_regret = immediate_total - best_other
        else:
            base_regret = -1.0

        effective_regret = base_regret + beta * urgency

        if effective_regret < wait_threshold:
            if (effective_regret < best_effective_regret or
                (effective_regret == best_effective_regret and immediate_total < best_immediate_total)):
                best_effective_regret = effective_regret
                best_immediate_total = immediate_total
                best_candidate = i

    return best_candidate