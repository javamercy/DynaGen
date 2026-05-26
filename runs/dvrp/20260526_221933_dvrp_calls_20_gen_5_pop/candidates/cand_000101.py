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
    
    # distances from current position and depot to all available customers
    dist_current = np.linalg.norm(available_customers - current_position, axis=1)
    dist_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    
    # find other trucks (excluding this truck if present)
    mask = np.all(truck_positions == current_position, axis=1)
    if mask.any():
        other_trucks = truck_positions[~mask]
    else:
        other_trucks = truck_positions
    
    if len(other_trucks) == 0:
        # single truck: simple score
        score = dist_current + 0.3 * dist_depot
        best_idx = np.argmin(score)
        return int(best_idx)
    
    # nearest distance to any other truck for each customer
    diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
    dist_other = np.linalg.norm(diff, axis=2)
    nearest_other = np.min(dist_other, axis=1)
    
    # waiting condition: if all customers are far from this truck relative to other trucks
    wait_threshold = 1.2
    if np.all(dist_current > wait_threshold * nearest_other):
        return None
    
    # compute depot coefficient based on current truck's distance to depot relative to fleet average
    curr_dist_to_depot = np.linalg.norm(current_position - depot_position)
    other_dist_to_depot = np.linalg.norm(other_trucks - depot_position, axis=1)
    avg_other_dist = np.mean(other_dist_to_depot)
    
    ratio = curr_dist_to_depot / (avg_other_dist + 1e-8)
    # base coefficient between 0.1 and 0.5 based on ratio
    base_coefficient = np.clip(0.2 + 0.2 * ratio, 0.1, 0.5)
    
    # time factor: increase depot pressure as time passes
    # estimate total time as twice the max distance from depot among available customers
    max_dist_depot = np.max(dist_depot) if len(dist_depot) > 0 else 1.0
    estimated_total_time = 2.0 * max_dist_depot + 1e-8
    time_factor = 1.0 + current_time / estimated_total_time
    time_factor = min(time_factor, 2.0)  # cap to avoid extreme values
    
    depot_coefficient = base_coefficient * time_factor
    
    score = dist_current - nearest_other + depot_coefficient * dist_depot
    best_idx = np.argmin(score)
    return int(best_idx)