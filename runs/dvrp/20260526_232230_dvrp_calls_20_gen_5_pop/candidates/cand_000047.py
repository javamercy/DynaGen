import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None

    # Compute current max return among all trucks (including this one if it stays)
    current_return_own = np.linalg.norm(current_position - depot_position)
    mask = ~np.all(truck_positions == current_position, axis=1)
    other_positions = truck_positions[mask]
    if other_positions.shape[0] > 0:
        other_returns = np.linalg.norm(other_positions - depot_position, axis=1)
        max_other_return = np.max(other_returns)
        min_other_return = np.min(other_returns)
    else:
        max_other_return = -np.inf
        min_other_return = np.inf

    current_max_return = max(current_return_own, max_other_return)

    best_idx = None
    best_score = float('inf')
    best_new_return = float('inf')

    for i, customer in enumerate(available_customers):
        d_truck_cust = np.linalg.norm(current_position - customer)
        d_cust_depot = np.linalg.norm(customer - depot_position)
        new_return = d_truck_cust + d_cust_depot
        new_max = max(new_return, max_other_return)
        # Fleet imbalance: difference between new max and what would be the new min
        # The min among trucks after assignment: min( new_return, min_other_return )
        new_min = min(new_return, min_other_return)
        imbalance = new_max - new_min
        # Score: new_max + small penalty for imbalance
        score = new_max + 0.1 * imbalance
        if (score < best_score) or (score == best_score and new_return < best_new_return):
            best_score = score
            best_new_return = new_return
            best_new_max = new_max
            best_idx = i

    # Wait if the best new max is more than 10% above current max return
    if best_new_max > current_max_return * 1.1:
        return None
    else:
        return best_idx