import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.size == 0:
        return None

    n_trucks = truck_positions.shape[0]
    dists = np.linalg.norm(truck_positions - current_position, axis=1)
    cur_idx = np.argmin(dists)

    # Baseline TTT if current truck returns directly to depot
    current_to_depot = np.linalg.norm(current_position - depot_position)
    other_times = []
    for j in range(n_trucks):
        if j == cur_idx:
            continue
        if available_customers.size > 0:
            dists_to_cust = np.linalg.norm(available_customers - truck_positions[j], axis=1)
            nearest = available_customers[np.argmin(dists_to_cust)]
            time = np.linalg.norm(truck_positions[j] - nearest) + np.linalg.norm(nearest - depot_position)
        else:
            time = np.linalg.norm(truck_positions[j] - depot_position)
        other_times.append(time)
    baseline_ttt = max(current_to_depot, max(other_times) if other_times else current_to_depot)

    n_customers = available_customers.shape[0]
    candidate_ttt = np.full(n_customers, np.inf)

    for i in range(n_customers):
        customer = available_customers[i]
        current_time = np.linalg.norm(current_position - customer) + np.linalg.norm(customer - depot_position)
        remaining = np.delete(available_customers, i, axis=0)
        other_times = []
        for j in range(n_trucks):
            if j == cur_idx:
                continue
            if remaining.size > 0:
                dists_to_rem = np.linalg.norm(remaining - truck_positions[j], axis=1)
                nearest = remaining[np.argmin(dists_to_rem)]
                time = np.linalg.norm(truck_positions[j] - nearest) + np.linalg.norm(nearest - depot_position)
            else:
                time = np.linalg.norm(truck_positions[j] - depot_position)
            other_times.append(time)
        candidate_ttt[i] = max(current_time, max(other_times) if other_times else current_time)

    best_idx = int(np.argmin(candidate_ttt))

    if n_trucks == 1:
        return best_idx
    else:
        if candidate_ttt[best_idx] < baseline_ttt - 1e-9:
            return best_idx
        else:
            return None