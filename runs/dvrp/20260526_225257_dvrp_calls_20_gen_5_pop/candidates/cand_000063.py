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

    # own_score: current->customer + customer->depot
    current_to_customer = np.linalg.norm(available_customers - current_position, axis=1)
    customer_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    own_score = current_to_customer + customer_to_depot

    # identify other trucks (exclude self)
    mask = np.all(np.abs(truck_positions - current_position) < 1e-8, axis=1)
    other_trucks = truck_positions[~mask]
    n_other = len(other_trucks)

    if n_other == 0:
        # only truck: just pick best own_score
        return int(np.argmin(own_score))

    # fleet balance: distance to centroid of other trucks
    centroid = np.mean(other_trucks, axis=0)
    dist_to_centroid = np.linalg.norm(available_customers - centroid, axis=1)

    # reduction in max depot distance
    current_max_depot = np.max(np.linalg.norm(truck_positions - depot_position, axis=1))
    other_max_depot = np.max(np.linalg.norm(other_trucks - depot_position, axis=1))
    # new_max if this truck goes to customer then depot
    new_max = np.maximum(other_max_depot, own_score)
    reduction = current_max_depot - new_max  # positive if reduces max

    # final score
    alpha = 0.2
    beta = 0.1
    score = own_score - alpha * dist_to_centroid - beta * reduction

    # compute regret based on own_score (simpler)
    best_idx = np.argmin(own_score)
    best_val = own_score[best_idx]
    second_best_val = np.partition(own_score, 1)[1]
    regret = second_best_val - best_val

    # waiting condition: not farthest, >2 customers, and regret small
    this_depot_dist = np.linalg.norm(current_position - depot_position)
    max_other_depot = np.max(np.linalg.norm(other_trucks - depot_position, axis=1)) if n_other > 0 else 0
    is_farthest = this_depot_dist >= max_other_depot
    if (not is_farthest) and (len(available_customers) > 2) and (regret < 0.1 * best_val):
        return None

    best_score_idx = np.argmin(score)
    return int(best_score_idx)