import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    n_trucks = len(truck_positions)
    n_available = len(available_customers)

    # Current truck's direct depot distance
    current_depot_dist = dist(current_position, depot_position)

    best_candidate = None
    best_regret = float('inf')
    best_cust_depot = float('inf')

    for i, cust in enumerate(available_customers):
        cust_depot = dist(cust, depot_position)
        # Incremental cost for current truck to serve customer then depot vs go direct
        inc_self = dist(current_position, cust) + cust_depot - current_depot_dist

        # Best incremental cost among other trucks
        best_other_inc = float('inf')
        for j, pos in enumerate(truck_positions):
            if np.array_equal(pos, current_position):
                continue
            other_depot = dist(pos, depot_position)
            inc_other = dist(pos, cust) + cust_depot - other_depot
            if inc_other < best_other_inc:
                best_other_inc = inc_other

        if n_trucks == 1:
            regret = -1.0  # always advantageous
        else:
            regret = inc_self - best_other_inc

        # Waiting threshold: adaptive based on current truck's depot distance and density
        density = n_available / max(n_trucks, 1)
        alpha = 0.1 + 0.4 * min(density / 5.0, 1.0)
        wait_threshold = alpha * current_depot_dist

        if regret < wait_threshold:
            if regret < best_regret or (np.isclose(regret, best_regret) and cust_depot < best_cust_depot):
                best_regret = regret
                best_cust_depot = cust_depot
                best_candidate = i

    # If no candidate satisfies regret condition, return None (wait)
    return best_candidate