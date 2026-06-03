import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    # distances from current truck to each customer
    dist_current_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    # distances from each customer to depot
    dist_cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    tour = dist_current_to_cust + dist_cust_to_depot
    # best other truck's tour for each customer
    best_other_tour = np.full(len(available_customers), np.inf)
    for pos in truck_positions:
        dist_other_to_cust = np.linalg.norm(available_customers - pos, axis=1)
        other_tour = dist_other_to_cust + dist_cust_to_depot
        best_other_tour = np.minimum(best_other_tour, other_tour)
    advantage = best_other_tour - tour
    penalty = 0.5
    combined = advantage - penalty * tour
    # if no positive combined score, wait
    if np.all(combined <= 0):
        return None
    best_idx = int(np.argmax(combined))
    return best_idx