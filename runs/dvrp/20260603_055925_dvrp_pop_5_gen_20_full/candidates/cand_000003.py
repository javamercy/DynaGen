import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.size == 0:
        return None

    # Identify current truck index
    dists = np.linalg.norm(truck_positions - current_position, axis=1)
    cur_idx = np.argmin(dists)

    n_customers = available_customers.shape[0]
    est_ttt = np.full(n_customers, np.inf)

    for i in range(n_customers):
        # Simulate that current truck serves customer i
        customer = available_customers[i]
        remaining = np.delete(available_customers, i, axis=0)  # customers left after this move

        costs = []
        for j in range(truck_positions.shape[0]):
            if j == cur_idx:
                # Current truck path: current->customer->nearest remaining->depot
                pos = current_position
                first_dist = np.linalg.norm(pos - customer)
                if remaining.size > 0:
                    # find nearest remaining to customer
                    dists_to_c = np.linalg.norm(remaining - customer, axis=1)
                    nearest_idx = np.argmin(dists_to_c)
                    nearest_cust = remaining[nearest_idx]
                    second_dist = np.linalg.norm(customer - nearest_cust)
                    return_dist = np.linalg.norm(nearest_cust - depot_position)
                    cost = first_dist + second_dist + return_dist
                else:
                    return_dist = np.linalg.norm(customer - depot_position)
                    cost = first_dist + return_dist
            else:
                # Other truck: from its current position to nearest remaining, then to depot
                pos = truck_positions[j]
                if remaining.size > 0:
                    dists_to_remaining = np.linalg.norm(remaining - pos, axis=1)
                    nearest_idx = np.argmin(dists_to_remaining)
                    nearest_cust = remaining[nearest_idx]
                    cost = np.linalg.norm(pos - nearest_cust) + np.linalg.norm(nearest_cust - depot_position)
                else:
                    cost = np.linalg.norm(pos - depot_position)
            costs.append(cost)
        est_ttt[i] = max(costs)

    best_idx = int(np.argmin(est_ttt))
    return best_idx