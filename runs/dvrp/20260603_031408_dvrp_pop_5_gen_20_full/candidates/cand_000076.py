import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    avg_truck_depot_dist = np.mean(np.linalg.norm(truck_positions - depot_position, axis=1))
    current_depot_dist = np.linalg.norm(current_position - depot_position)
    best_idx = None
    best_score = -np.inf
    w_isolation = 0.7
    w_balance = 0.3
    for i, cust in enumerate(available_customers):
        dist_to_truck = np.linalg.norm(current_position - cust)
        dist_to_depot = np.linalg.norm(depot_position - cust)
        min_other = np.inf
        for j, pos in enumerate(truck_positions):
            if np.allclose(pos, current_position):
                continue
            d = np.linalg.norm(pos - cust)
            if d < min_other:
                min_other = d
        if min_other == np.inf:
            min_other = 0.0
        balance_term = current_depot_dist - avg_truck_depot_dist
        score = dist_to_depot - dist_to_truck - w_isolation * min_other - w_balance * balance_term
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx