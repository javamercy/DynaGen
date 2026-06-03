import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    
    # Find index of active truck (the truck at current_position)
    active_idx = -1
    for idx, pos in enumerate(truck_positions):
        if np.linalg.norm(pos - current_position) < 1e-9:
            active_idx = idx
            break
    if active_idx == -1:
        active_idx = 0  # fallback
    
    def simulate_remaining(customers, start_positions, depot):
        # customers: (N,2) array of unassigned customers
        # start_positions: list of (2,) arrays for each truck (M trucks)
        n_trucks = len(start_positions)
        if len(customers) == 0:
            # each truck returns directly from start position
            return max(np.linalg.norm(start_positions[i] - depot) for i in range(n_trucks))
        
        # state per truck: last position and total distance traveled so far
        last_pos = [pos.copy() for pos in start_positions]
        total_dist = [0.0 for _ in range(n_trucks)]
        unassigned = [list(cust) for cust in customers]  # list of lists for easy removal
        
        while unassigned:
            best_increase = float('inf')
            best_truck = -1
            best_cust_idx = -1
            # For each unassigned customer and each truck, compute increase in makespan if assigned
            for j, cust in enumerate(unassigned):
                for t in range(n_trucks):
                    if total_dist[t] == 0.0 and len(unassigned) == 1 and np.allclose(last_pos[t], cust):
                        # already at customer? but this is rare
                        inc = np.linalg.norm(cust - depot) - np.linalg.norm(last_pos[t] - depot)
                    else:
                        inc = np.linalg.norm(last_pos[t] - cust) + np.linalg.norm(cust - depot) - np.linalg.norm(last_pos[t] - depot if total_dist[t] > 0 else start_positions[t] - depot)
                        if total_dist[t] == 0:
                            inc = np.linalg.norm(start_positions[t] - cust) + np.linalg.norm(cust - depot) - np.linalg.norm(start_positions[t] - depot)
                        else:
                            inc = np.linalg.norm(last_pos[t] - cust) + np.linalg.norm(cust - depot) - np.linalg.norm(last_pos[t] - depot)
                    # compute new total for this truck
                    new_truck_total = total_dist[t] + inc
                    # new makespan if this assignment chosen: max(current other totals, new_truck_total)
                    current_makespan = max(total_dist)
                    # But we need to consider that other trucks remain same
                    # Actually after assignment, the makespan becomes max(max(total_dist for all trucks except t), new_truck_total)
                    # Since we compare pairs, we can compute directly
                    candidate_makespan = max(max(total_dist[:t] + total_dist[t+1:]), new_truck_total)
                    if candidate_makespan < best_increase:  # we want minimum makespan
                        best_increase = candidate_makespan
                        best_truck = t
                        best_cust_idx = j
            # assign best customer to best truck
            cust_assigned = unassigned.pop(best_cust_idx)
            # update truck's last_pos and total_dist
            if total_dist[best_truck] == 0:
                travel = np.linalg.norm(start_positions[best_truck] - cust_assigned)
            else:
                travel = np.linalg.norm(last_pos[best_truck] - cust_assigned)
            total_dist[best_truck] += travel
            last_pos[best_truck] = np.array(cust_assigned)
        # after all customers assigned, add return to depot
        final_dist = [total_dist[i] + np.linalg.norm(last_pos[i] - depot) for i in range(n_trucks)]
        return max(final_dist)
    
    best_candidate = None
    best_ttt = float('inf')
    
    # Evaluate None (wait)
    other_truck_positions = [truck_positions[i] for i in range(len(truck_positions)) if i != active_idx]
    if len(other_truck_positions) == 0:
        ttt_none = np.linalg.norm(current_position - depot_position)
    else:
        ttt_other = simulate_remaining(available_customers, other_truck_positions, depot_position)
        ttt_none = max(ttt_other, np.linalg.norm(current_position - depot_position))
    best_ttt = ttt_none
    best_candidate = None
    
    # Evaluate each available customer
    for i, cust in enumerate(available_customers):
        # start positions for all trucks: active truck now at customer, others unchanged
        start_positions = [cust] + [truck_positions[j] for j in range(len(truck_positions)) if j != active_idx]
        remaining = np.delete(available_customers, i, axis=0)
        ttt = simulate_remaining(remaining, start_positions, depot_position)
        if ttt < best_ttt:
            best_ttt = ttt
            best_candidate = i
    return best_candidate