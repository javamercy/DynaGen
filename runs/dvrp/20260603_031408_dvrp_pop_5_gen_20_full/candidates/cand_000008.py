import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    cur = current_position
    avail = available_customers
    n = len(avail)
    # find index of current truck in truck_positions (if multiple close, take first)
    current_truck_idx = None
    for idx, pos in enumerate(truck_positions):
        if np.allclose(pos, cur, atol=1e-6):
            current_truck_idx = idx
            break
    best_idx = None
    best_score = -np.inf
    for i in range(n):
        cust = avail[i]
        d_cur = np.linalg.norm(cur - cust)
        # nearest neighbor distance among other customers
        if n == 1:
            min_nn = 0.0
        else:
            min_nn = np.inf
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(cust - avail[j])
                    if d < min_nn:
                        min_nn = d
        # distance to nearest other truck
        min_dist_other = np.inf
        for idx, pos in enumerate(truck_positions):
            if current_truck_idx is not None and idx == current_truck_idx:
                continue
            d = np.linalg.norm(cust - pos)
            if d < min_dist_other:
                min_dist_other = d
        # If no other trucks, ignore term (set to 0)
        if np.isinf(min_dist_other):
            min_dist_other = 0.0
        score = min_nn - d_cur - min_dist_other
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx