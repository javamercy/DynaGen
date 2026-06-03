import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    n_trucks = len(truck_positions)
    # Find index of current truck
    idx_current = None
    for i, pos in enumerate(truck_positions):
        if np.allclose(pos, current_position):
            idx_current = i
            break
    if idx_current is None:
        idx_current = 0  # fallback, should not happen

    self_dist = dist(current_position, depot_position)

    # Single truck case: always pick best customer
    if n_trucks == 1:
        best_idx = None
        best_total = float('inf')
        for i, cust in enumerate(available_customers):
            total = dist(current_position, cust) + dist(cust, depot_position)
            if total < best_total:
                best_total = total
                best_idx = i
        return best_idx

    # Compute other trucks' distances to depot
    other_dists = []
    for j, pos in enumerate(truck_positions):
        if j != idx_current:
            other_dists.append(dist(pos, depot_position))
    avg_other_dist = np.mean(other_dists) if other_dists else 0.0

    best_negative_regret = float('inf')
    best_negative_idx = None
    best_negative_immediate = float('inf')
    best_positive_regret = float('inf')
    best_positive_idx = None
    best_positive_immediate = float('inf')

    for i, cust in enumerate(available_customers):
        cust_return = dist(cust, depot_position)
        immediate = dist(current_position, cust) + cust_return
        # Compute best other time
        best_other = float('inf')
        for j, pos in enumerate(truck_positions):
            if j == idx_current:
                continue
            deferred = dist(pos, cust) + cust_return
            if deferred < best_other:
                best_other = deferred
        regret = immediate - best_other
        if regret < 0:
            if regret < best_negative_regret or (regret == best_negative_regret and immediate < best_negative_immediate):
                best_negative_regret = regret
                best_negative_immediate = immediate
                best_negative_idx = i
        else:
            if regret < best_positive_regret or (regret == best_positive_regret and immediate < best_positive_immediate):
                best_positive_regret = regret
                best_positive_immediate = immediate
                best_positive_idx = i

    if best_negative_idx is not None:
        return best_negative_idx

    # No negative regret
    if self_dist < 1e-6:  # at depot, always pick
        return best_positive_idx

    # Adaptive threshold (modified parameters)
    ratio = avg_other_dist / self_dist if self_dist > 1e-6 else 2.0
    ratio = np.clip(ratio, 0.6, 1.8)
    alpha = 0.25 * ratio
    threshold = alpha * self_dist

    if best_positive_idx is not None and best_positive_regret < threshold:
        return best_positive_idx
    else:
        return None