import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    n_trucks = len(truck_positions)
    if n_trucks == 1:
        # single truck: go to nearest customer
        distances = np.linalg.norm(available_customers - current_position, axis=1)
        return int(np.argmin(distances))

    # find index of current truck
    current_idx = int(np.argmin(np.linalg.norm(truck_positions - current_position, axis=1)))

    # precompute distances to depot for all trucks
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max_depot = np.max(dist_to_depot)

    # first pass: look for negative regret
    best_negative_idx = None
    best_negative_regret = float('inf')
    best_negative_immediate = float('inf')

    for i, cust in enumerate(available_customers):
        cust_depot = dist(cust, depot_position)
        current_cost = dist(current_position, cust) + cust_depot

        # best other truck cost
        best_other = float('inf')
        for j, pos in enumerate(truck_positions):
            if j == current_idx:
                continue
            other_cost = dist(pos, cust) + cust_depot
            if other_cost < best_other:
                best_other = other_cost
        regret = best_other - current_cost

        if regret < 0 and (regret < best_negative_regret or (regret == best_negative_regret and current_cost < best_negative_immediate)):
            best_negative_regret = regret
            best_negative_immediate = current_cost
            best_negative_idx = i

    if best_negative_idx is not None:
        return best_negative_idx

    # no negative regret: compute composite score for each customer
    best_score = float('inf')
    best_candidate = None
    best_regret = float('inf')

    for i, cust in enumerate(available_customers):
        cust_depot = dist(cust, depot_position)
        current_cost = dist(current_position, cust) + cust_depot

        best_other = float('inf')
        for j, pos in enumerate(truck_positions):
            if j == current_idx:
                continue
            other_cost = dist(pos, cust) + cust_depot
            if other_cost < best_other:
                best_other = other_cost
        regret = best_other - current_cost

        # penalty for increasing max depot distance
        new_max = max(current_cost, np.max(np.delete(dist_to_depot, current_idx)))
        penalty = 0.1 * (new_max - current_max_depot)
        score = regret - penalty

        if score < best_score or (score == best_score and regret < best_regret):
            best_score = score
            best_regret = regret
            best_candidate = i

    if best_candidate is None:
        return None

    # adaptive waiting threshold
    n_available = len(available_customers)
    density = n_available / max(n_trucks, 1)
    alpha = 0.1 + 0.4 * (density / 5.0)
    alpha = min(max(alpha, 0.1), 0.5)
    dist_to_depot_current = dist(current_position, depot_position)
    wait_threshold = alpha * dist_to_depot_current

    if best_score < wait_threshold:
        return best_candidate
    else:
        return None