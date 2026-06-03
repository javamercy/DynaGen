import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None

    # Compute distances
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    cur_to_depot = np.linalg.norm(current_position - depot_position)
    cur_to_cust = np.linalg.norm(available_customers - current_position, axis=1)

    # Savings score: cur_to_depot + cust_to_depot - cur_to_cust
    savings = cur_to_depot + cust_to_depot - cur_to_cust
    lambda_ = 0.1
    scores = savings - lambda_ * cust_to_depot

    # Active cost for tie-breaking
    active_cost = cur_to_cust + cust_to_depot

    # Best index: highest score, then smallest active_cost
    best_idx = np.lexsort((active_cost, -scores))[0]
    return int(best_idx)