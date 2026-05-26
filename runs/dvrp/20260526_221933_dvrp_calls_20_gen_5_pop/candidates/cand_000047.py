import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if len(available_customers) == 0:
        return None
    # distances from customers to all trucks
    dist_mat = np.linalg.norm(available_customers[:, np.newaxis, :] - truck_positions[np.newaxis, :, :], axis=2)
    # index of current truck
    current_truck_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    nearest_truck_idx = np.argmin(dist_mat, axis=1)  # per customer, nearest truck index
    # customers for which current truck is nearest
    my_customers_mask = nearest_truck_idx == current_truck_idx
    if np.any(my_customers_mask):
        # among my customers, choose the one minimizing distance to customer + customer to depot
        dist_current = np.linalg.norm(available_customers - current_position, axis=1)
        cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
        cost = dist_current + cust_to_depot
        cost[~my_customers_mask] = np.inf
        best_idx = np.argmin(cost)
        return int(best_idx)
    else:
        # fallback: choose the most isolated customer (largest min distance to any truck)
        min_dist = np.min(dist_mat, axis=1)
        best_idx = np.argmax(min_dist)
        return int(best_idx)