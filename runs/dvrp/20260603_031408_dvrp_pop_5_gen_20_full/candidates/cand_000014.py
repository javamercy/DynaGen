import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    n = len(available_customers)
    lambda_depot = 0.5 * np.exp(-n / 20.0)
    best_idx = None
    best_score = -np.inf
    cur = current_position
    depot = depot_position
    for i in range(n):
        d_cur = np.linalg.norm(cur - available_customers[i])
        d_depot = np.linalg.norm(depot - available_customers[i])
        if n == 1:
            min_nn = 0.0
        else:
            min_nn = np.inf
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(available_customers[i] - available_customers[j])
                    if d < min_nn:
                        min_nn = d
        score = min_nn - d_cur - lambda_depot * d_depot
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx