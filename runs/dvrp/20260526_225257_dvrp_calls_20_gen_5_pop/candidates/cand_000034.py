import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    n_available = len(available_customers)
    if n_available == 0:
        return None

    n_trucks = truck_positions.shape[0]

    # Identify deciding truck index
    dist_to_current = np.linalg.norm(truck_positions - current_position, axis=1)
    deciding_truck_idx = np.argmin(dist_to_current)

    depot = depot_position
    cust_to_depot = np.linalg.norm(available_customers - depot, axis=1)
    # Scale for tolerance: mean distance from depot to available customers
    mean_depot_dist = np.mean(cust_to_depot) if n_available > 0 else 0.0

    regrets = np.empty(n_available)
    for i in range(n_available):
        cust = available_customers[i]
        d_curr_cust = np.linalg.norm(current_position - cust)
        cost_now = d_curr_cust + cust_to_depot[i]

        best_other = float('inf')
        for j in range(n_trucks):
            if j == deciding_truck_idx:
                continue
            d_truck_cust = np.linalg.norm(truck_positions[j] - cust)
            cost_other = d_truck_cust + cust_to_depot[i]
            if cost_other < best_other:
                best_other = cost_other

        if n_trucks == 1:
            regrets[i] = -float('inf')
        else:
            regrets[i] = cost_now - best_other

    # Compute tolerance based on proximity to depot and number of available customers
    dist_curr_depot = np.linalg.norm(current_position - depot)
    # Tolerance increases when near depot or few customers
    tolerance = 0.0
    if n_available <= 3:
        tolerance = 0.05 * mean_depot_dist
    if dist_curr_depot < 0.1 * mean_depot_dist:
        tolerance = max(tolerance, 0.05 * mean_depot_dist)

    # Candidates: regret <= tolerance
    candidates = [i for i in range(n_available) if regrets[i] <= tolerance]

    if not candidates:
        # Fallback: choose customer with minimal regret (best positive regret)
        best_fallback = int(np.argmin(regrets))
        return best_fallback

    # Among candidates, use score as before
    beta = 1.0 * (n_available / (n_available + 5))
    tf = min(1.0, current_time / 100.0)

    best_score = float('inf')
    best_index = None
    for i in candidates:
        cust = available_customers[i]
        d_curr_cust = np.linalg.norm(current_position - cust)
        isolation = float('inf')
        for j in range(n_trucks):
            if j == deciding_truck_idx:
                continue
            d = np.linalg.norm(truck_positions[j] - cust)
            if d < isolation:
                isolation = d
        if n_trucks == 1:
            isolation = 0.0
        score = d_curr_cust + (1 + tf) * cust_to_depot[i] - beta * isolation
        if score < best_score:
            best_score = score
            best_index = i

    return best_index