import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    n_trucks = len(truck_positions)
    if n_trucks == 1:
        distances = np.linalg.norm(available_customers - current_position, axis=1)
        return int(np.argmin(distances))
    # Find current truck index
    current_truck_idx = int(np.argmin(np.linalg.norm(truck_positions - current_position, axis=1)))
    # Compute current distance to depot
    current_dist_to_depot = np.linalg.norm(current_position - depot_position)
    # Compute max distance to depot among other trucks
    other_max = 0.0
    for j, pos in enumerate(truck_positions):
        if j == current_truck_idx:
            continue
        d = np.linalg.norm(pos - depot_position)
        if d > other_max:
            other_max = d
    best_improvement = -float('inf')
    best_idx = None
    best_current_total = float('inf')
    for i, cust in enumerate(available_customers):
        d_curr = np.linalg.norm(current_position - cust)
        cust_depot = np.linalg.norm(cust - depot_position)
        current_total = d_curr + cust_depot
        max_if_go = max(current_total, other_max)
        max_if_wait = max(current_dist_to_depot, other_max)
        improvement = max_if_wait - max_if_go
        # Tie-break: prefer smaller current_total
        if (improvement > best_improvement) or (improvement == best_improvement and current_total < best_current_total):
            best_improvement = improvement
            best_idx = i
            best_current_total = current_total
    if best_improvement > 0:
        return best_idx
    else:
        return None