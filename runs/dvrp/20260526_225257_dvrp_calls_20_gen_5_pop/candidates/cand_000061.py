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
    # identify deciding truck index
    dist_to_current = np.linalg.norm(truck_positions - current_position, axis=1)
    deciding_truck_idx = np.argmin(dist_to_current)

    # distances from each available customer to depot
    dist_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)

    # compute regrets
    regrets = np.empty(n_available)
    for i, cust in enumerate(available_customers):
        d_curr = np.linalg.norm(current_position - cust)
        cost_now = current_time + d_curr + dist_to_depot[i]
        best_other = float('inf')
        for j in range(n_trucks):
            if j == deciding_truck_idx:
                continue
            d_truck = np.linalg.norm(truck_positions[j] - cust)
            cost_other = current_time + d_truck + dist_to_depot[i]
            if cost_other < best_other:
                best_other = cost_other
        if n_trucks == 1:
            regrets[i] = -float('inf')
        else:
            regrets[i] = cost_now - best_other

    # filter non-positive regret candidates
    candidates = [i for i in range(n_available) if regrets[i] <= 0]
    if not candidates:
        return None

    # lightweight waiting evaluation using top 3 farthest customers
    L = min(3, n_available)
    sorted_by_dist = np.argsort(dist_to_depot)[::-1]
    topL_indices = sorted_by_dist[:L].tolist()

    # helper to compute best time for a truck to serve a set of customers and return to depot
    # for each truck, we consider it serves exactly one customer from the set (the best one for that truck)
    def estimate_max_other_time(truck_indices, customer_indices):
        if len(customer_indices) == 0:
            return 0.0
        max_time = 0.0
        for t_idx in truck_indices:
            truck_pos = truck_positions[t_idx]
            best_time = float('inf')
            for c_idx in customer_indices:
                cust = available_customers[c_idx]
                t = np.linalg.norm(truck_pos - cust) + np.linalg.norm(cust - depot_position)
                if t < best_time:
                    best_time = t
            if best_time > max_time:
                max_time = best_time
        return max_time

    other_truck_indices = [i for i in range(n_trucks) if i != deciding_truck_idx]

    # waiting: deciding truck does nothing, other trucks serve top L customers (one each)
    waiting_other_time = estimate_max_other_time(other_truck_indices, topL_indices)
    # deciding truck will eventually return, assume at least direct to depot
    waiting_deciding_time = np.linalg.norm(current_position - depot_position)
    waiting_ttt = current_time + max(waiting_other_time, waiting_deciding_time)

    best_candidate = None
    best_candidate_ttt = float('inf')
    for cand_idx in candidates:
        # candidate set: top L excluding candidate itself
        remaining = [idx for idx in topL_indices if idx != cand_idx]
        # deciding truck serves candidate
        deciding_time = np.linalg.norm(current_position - available_customers[cand_idx]) + \
                        np.linalg.norm(available_customers[cand_idx] - depot_position)
        other_time = estimate_max_other_time(other_truck_indices, remaining)
        candidate_ttt = current_time + max(deciding_time, other_time)
        if candidate_ttt < best_candidate_ttt:
            best_candidate_ttt = candidate_ttt
            best_candidate = cand_idx

    if waiting_ttt < best_candidate_ttt:
        return None
    else:
        return best_candidate