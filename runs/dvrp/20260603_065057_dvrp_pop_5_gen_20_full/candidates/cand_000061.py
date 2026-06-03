import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    # identify the deciding truck
    diffs = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diffs)
    # waiting case: current truck stays put
    current_wait_work = np.linalg.norm(current_position - depot_position)
    other_wait_work = max(np.linalg.norm(truck_positions[j] - depot_position) for j in range(len(truck_positions)) if j != current_idx)
    wait_max = max(current_wait_work, other_wait_work)
    best_score = -np.inf
    best_idx = -1
    best_this_cost = None
    for i, cust in enumerate(available_customers):
        this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        other_max = max(np.linalg.norm(truck_positions[j] - depot_position) for j in range(len(truck_positions)) if j != current_idx)
        max_work = max(this_cost, other_max)
        score = -max_work  # we want to minimize max_work
        # tie-break by smaller this_cost
        if score > best_score or (score == best_score and (best_this_cost is None or this_cost < best_this_cost)):
            best_score = score
            best_idx = i
            best_this_cost = this_cost
    # compare with waiting
    if best_score > -wait_max:
        return best_idx
    else:
        return None