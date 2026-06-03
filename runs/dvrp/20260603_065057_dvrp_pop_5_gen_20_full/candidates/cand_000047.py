import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None

    # distances from current truck to each customer
    d_curr = np.linalg.norm(available_customers - current_position, axis=1)
    # distances from each customer to depot
    d_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    # total travel if current truck serves customer and returns
    T_curr = d_curr + d_depot

    n_trucks = truck_positions.shape[0]
    if n_trucks == 1:
        best_idx = np.argmin(T_curr)
        return int(best_idx)

    # identify current truck index
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diff)

    # compute total travel for each customer if served by each other truck
    # shape: (n_trucks-1, n_cust)
    other_indices = [i for i in range(n_trucks) if i != current_idx]
    other_trucks = truck_positions[other_indices]
    # distances from other trucks to customers
    d_other = np.linalg.norm(other_trucks[:, np.newaxis, :] - available_customers, axis=2)  # (n_other, n_cust)
    T_other = d_other + d_depot  # each row: total travel for an other truck
    # best other travel per customer
    T_best_other = np.min(T_other, axis=0)

    # customers where current truck is at least as good as best other
    mask = T_curr <= T_best_other + 1e-9
    if np.any(mask):
        # among those, pick the one with smallest T_curr
        best_idx = np.argmin(np.where(mask, T_curr, np.inf))
        return int(best_idx)
    else:
        return None