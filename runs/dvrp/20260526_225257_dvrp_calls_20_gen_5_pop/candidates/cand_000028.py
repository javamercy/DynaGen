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

    # Identify the deciding truck index
    dist_to_current = np.linalg.norm(truck_positions - current_position, axis=1)
    deciding_truck_idx = np.argmin(dist_to_current)

    depot = depot_position
    cust_to_depot = np.linalg.norm(available_customers - depot, axis=1)

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

    candidates = [i for i in range(n_available) if regrets[i] <= 0]
    if not candidates:
        return None

    # Decaying isolation weight
    beta = 1.0 * (n_available / (n_available + 5))
    # Time-dependent depot weight factor
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