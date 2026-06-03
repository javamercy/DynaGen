import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    # distances from active truck to each customer
    d_active = np.linalg.norm(available_customers - current_position, axis=1)
    # distances from customers to depot
    d_cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    avg_depot_dist = np.mean(d_cust_to_depot)
    # find active truck index
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    other_indices = [i for i in range(len(truck_positions)) if i != active_idx]
    if len(other_indices) > 0:
        other_positions = truck_positions[other_indices]
        d_other_min = np.min(np.linalg.norm(available_customers[:, None] - other_positions[None], axis=2), axis=1)
        ratio = np.where(d_active > 1e-9, d_other_min / d_active, np.inf)
    else:
        ratio = np.ones(len(d_active)) * 1e9  # only one truck, ratio large
    # select candidate by regret
    mask = ratio > 1.0  # active is closer than any other truck
    if np.any(mask):
        candidates = np.where(mask, ratio, -np.inf)
        best = np.argmax(candidates)
    else:
        # fallback to nearest neighbor
        best = np.argmin(d_active)
    # wait condition
    if d_active[best] > 2.0 * avg_depot_dist:
        return None
    return int(best)