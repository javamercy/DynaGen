import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    best_idx = None
    best_score = -np.inf
    cur = current_position
    avail = available_customers
    n = len(avail)
    for i in range(n):
        d_cur = np.linalg.norm(cur - avail[i])
        # nearest neighbor distance excluding i
        if n == 1:
            min_nn = 0.0
        else:
            min_nn = np.inf
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(avail[i] - avail[j])
                    if d < min_nn:
                        min_nn = d
        score = min_nn - d_cur
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx