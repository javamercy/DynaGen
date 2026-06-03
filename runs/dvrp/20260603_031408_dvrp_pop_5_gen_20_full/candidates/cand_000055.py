def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    dists_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # find index of current truck
    current_idx = None
    for j, pos in enumerate(truck_positions):
        if np.array_equal(pos, current_position):
            current_idx = j
            break
    if current_idx is not None:
        max_without_current = np.max(np.delete(dists_to_depot, current_idx))
    else:
        max_without_current = np.max(dists_to_depot)
    current_makespan = np.max(dists_to_depot)
    best_idx = None
    best_cost = float('inf')
    for i, cust in enumerate(available_customers):
        potential = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        new_makespan = max(potential, max_without_current)
        if new_makespan < best_cost:
            best_cost = new_makespan
            best_idx = i
    if best_cost > current_makespan:
        return None
    else:
        return best_idx