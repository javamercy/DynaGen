import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    n_trucks = truck_positions.shape[0]
    # find current truck index
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diff)
    # precompute other trucks' direct return times (to depot)
    other_etas = []
    for j in range(n_trucks):
        if j == current_idx:
            continue
        other_etas.append(np.linalg.norm(truck_positions[j] - depot_position))
    max_other = max(other_etas) if other_etas else 0.0
    best_score = -np.inf
    best_idx = -1
    best_current_eta = np.inf
    for i in range(available_customers.shape[0]):
        cust = available_customers[i]
        current_eta = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        max_eta = max(current_eta, max_other)
        # compute regret: difference between best other truck's cost and current truck's cost
        other_min = np.inf
        for j in range(n_trucks):
            if j == current_idx:
                continue
            other_cost = np.linalg.norm(truck_positions[j] - cust) + np.linalg.norm(cust - depot_position)
            if other_cost < other_min:
                other_min = other_cost
        regret = max(0.0, other_min - current_eta) if other_min != np.inf else 0.0
        score = -max_eta + 0.1 * regret
        if score > best_score or (score == best_score and current_eta < best_current_eta):
            best_score = score
            best_idx = i
            best_current_eta = current_eta
    return best_idx