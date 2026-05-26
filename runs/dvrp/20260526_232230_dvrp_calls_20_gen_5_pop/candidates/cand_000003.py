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
    
    # Find index of current truck (assume unique position)
    matches = np.where(np.all(truck_positions == current_position, axis=1))[0]
    if len(matches) == 0:
        raise ValueError("Current position not found in truck_positions")
    current_idx = matches[0]
    n_trucks = len(truck_positions)
    
    # Define simulation function
    def simulate_ttt(first_assignment):
        # first_assignment: (customer_idx, truck_idx) or None
        pos = truck_positions.copy()  # start positions for all trucks
        times = np.zeros(n_trucks)   # elapsed travel time per truck
        assigned = np.zeros(len(available_customers), dtype=bool)
        
        if first_assignment is not None:
            c_idx, t_idx = first_assignment
            assigned[c_idx] = True
            dist = np.linalg.norm(pos[t_idx] - available_customers[c_idx])
            times[t_idx] += dist
            pos[t_idx] = available_customers[c_idx]
        
        # Round-robin: each truck takes its nearest unassigned customer in turn
        while not np.all(assigned):
            for t_idx in range(n_trucks):
                best_dist = np.inf
                best_c = -1
                for c_idx in range(len(available_customers)):
                    if not assigned[c_idx]:
                        d = np.linalg.norm(pos[t_idx] - available_customers[c_idx])
                        if d < best_dist:
                            best_dist = d
                            best_c = c_idx
                if best_c != -1:
                    assigned[best_c] = True
                    times[t_idx] += best_dist
                    pos[t_idx] = available_customers[best_c]
        
        # Return to depot
        for t_idx in range(n_trucks):
            times[t_idx] += np.linalg.norm(pos[t_idx] - depot_position)
        return np.max(times)
    
    best_ttt = np.inf
    best_idx = None
    
    # Evaluate wait option
    ttt_wait = simulate_ttt(None)
    if ttt_wait < best_ttt:
        best_ttt = ttt_wait
        best_idx = None
    
    # Evaluate each customer
    for i in range(len(available_customers)):
        ttt = simulate_ttt((i, current_idx))
        if ttt < best_ttt:
            best_ttt = ttt
            best_idx = i
    
    return best_idx