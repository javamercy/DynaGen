import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # find index of the deciding truck
    truck_idx = None
    for i, pos in enumerate(truck_positions):
        if np.array_equal(pos, current_position):
            truck_idx = i
            break
    if truck_idx is None:
        truck_idx = 0  # fallback

    # distances from each truck to depot
    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)

    def compute_max_time(customer_index):
        cust = available_customers[customer_index]
        new_dist = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        new_times = truck_to_depot.copy()
        new_times[truck_idx] = new_dist
        return np.max(new_times)

    best_score = np.max(truck_to_depot)
    best_action = None

    for i in range(len(available_customers)):
        score = compute_max_time(i)
        if score < best_score:
            best_score = score
            best_action = i

    return best_action