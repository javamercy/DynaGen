import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    # Find index of current truck
    current_idx = np.where(np.all(np.isclose(truck_positions, current_position), axis=1))[0]
    if len(current_idx) == 0:
        # current_position not in truck_positions (shouldn't happen); treat separately
        other_returns = depot_dists
        current_direct = np.linalg.norm(current_position - depot_position)
        current_max = max(np.max(depot_dists), current_direct)
    else:
        current_idx = current_idx[0]
        other_returns = np.delete(depot_dists, current_idx)
        current_direct = depot_dists[current_idx]
        current_max = np.max(depot_dists)

    # Single truck case: always assign the customer with smallest immediate distance
    if len(other_returns) == 0:
        best_imm = float('inf')
        best_idx = None
        for i, cust in enumerate(available_customers):
            imm = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
            if imm < best_imm:
                best_imm = imm
                best_idx = i
        return best_idx

    # Multiple trucks
    other_max = np.max(other_returns)
    best_new_max = float('inf')
    best_imm = float('inf')
    best_idx = None
    for i, cust in enumerate(available_customers):
        imm = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        new_max = max(other_max, imm)
        if new_max < best_new_max or (new_max == best_new_max and imm < best_imm):
            best_new_max = new_max
            best_imm = imm
            best_idx = i

    if best_new_max <= current_max:
        return best_idx
    else:
        return None