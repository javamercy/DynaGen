import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None

    # identify index of current truck
    truck_idx = None
    for i, pos in enumerate(truck_positions):
        if np.array_equal(pos, current_position):
            truck_idx = i
            break
    if truck_idx is None:
        # fallback: closest truck
        dists = np.linalg.norm(truck_positions - current_position, axis=1)
        truck_idx = int(np.argmin(dists))

    # precompute costs: for each customer, cost if served by current truck vs best other truck
    n_cust = len(available_customers)
    n_trucks = len(truck_positions)

    # distances from current truck to each customer
    d_curr_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    # distances from each customer to depot
    d_cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    # cost if current truck serves: d_curr_to_cust + d_cust_to_depot
    cost_this = d_curr_to_cust + d_cust_to_depot

    # compute best cost from any other truck
    best_other_cost = np.full(n_cust, np.inf)
    for t in range(n_trucks):
        if t == truck_idx:
            continue
        d_truck_to_cust = np.linalg.norm(available_customers - truck_positions[t], axis=1)
        cost_other = d_truck_to_cust + d_cust_to_depot
        best_other_cost = np.minimum(best_other_cost, cost_other)

    regret = cost_this - best_other_cost

    # if no customer has negative or zero regret, wait
    if np.all(regret > 0):
        return None

    # among customers with regret <= 0, pick one with smallest cost_this
    candidates = np.where(regret <= 0)[0]
    best_idx = candidates[np.argmin(cost_this[candidates])]

    return int(best_idx)