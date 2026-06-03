import numpy as np
import heapq

def simulate_TTT(active_idx, truck_positions, depot, customers, active_assignment):
    n_trucks = len(truck_positions)
    # Mutable copies
    truck_pos = [truck_positions[i].copy() for i in range(n_trucks)]
    unserved = [c.copy() for c in customers]
    truck_dest = [None] * n_trucks
    truck_arrival = [None] * n_trucks
    last_service_time = [0.0] * n_trucks
    current_time = 0.0
    event_queue = []
    
    # Handle active truck assignment
    if active_assignment is not None:
        # Remove customer from unserved; assume index is valid
        dest = unserved.pop(active_assignment)
        truck_dest[active_idx] = dest
        dist = np.linalg.norm(truck_pos[active_idx] - dest)
        truck_arrival[active_idx] = current_time + dist
        heapq.heappush(event_queue, (truck_arrival[active_idx], active_idx))
    
    # Initial dispatch for idle trucks (including active if not assigned)
    for t in range(n_trucks):
        if truck_dest[t] is None and unserved:
            # Find nearest unserved customer
            dists = [np.linalg.norm(truck_pos[t] - c) for c in unserved]
            nearest_idx = np.argmin(dists)
            dest = unserved.pop(nearest_idx)
            truck_dest[t] = dest
            dist = dists[nearest_idx]
            truck_arrival[t] = current_time + dist
            heapq.heappush(event_queue, (truck_arrival[t], t))
    
    # Event loop
    while event_queue or unserved:
        if not event_queue and unserved:
            # All trucks idle, but customers remain (should not happen if unserved handled)
            # Dispatch all idle trucks immediately
            for t in range(n_trucks):
                if truck_dest[t] is None and unserved:
                    dists = [np.linalg.norm(truck_pos[t] - c) for c in unserved]
                    nearest_idx = np.argmin(dists)
                    dest = unserved.pop(nearest_idx)
                    truck_dest[t] = dest
                    dist = dists[nearest_idx]
                    truck_arrival[t] = current_time + dist
                    heapq.heappush(event_queue, (truck_arrival[t], t))
        # Process next event
        time, t_idx = heapq.heappop(event_queue)
        current_time = time
        # Truck arrives at destination
        truck_pos[t_idx] = truck_dest[t_idx].copy()
        last_service_time[t_idx] = current_time
        truck_dest[t_idx] = None
        # If customers remain, assign nearest
        if unserved:
            dists = [np.linalg.norm(truck_pos[t_idx] - c) for c in unserved]
            nearest_idx = np.argmin(dists)
            dest = unserved.pop(nearest_idx)
            truck_dest[t_idx] = dest
            dist = dists[nearest_idx]
            truck_arrival[t_idx] = current_time + dist
            heapq.heappush(event_queue, (truck_arrival[t_idx], t_idx))
    
    # All customers served, compute return times
    max_return = 0.0
    for t in range(n_trucks):
        return_dist = np.linalg.norm(truck_pos[t] - depot)
        return_time = last_service_time[t] + return_dist
        if return_time > max_return:
            max_return = return_time
    return max_return

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # Find index of active truck (position match)
    active_idx = np.where((truck_positions == current_position).all(axis=1))[0]
    if len(active_idx) == 0:
        # Fallback: nearest truck
        dists = np.linalg.norm(truck_positions - current_position, axis=1)
        active_idx = np.argmin(dists)
    else:
        active_idx = active_idx[0]
    
    best_ttt = float('inf')
    best_candidate = None
    # Evaluate each customer index and the wait option (None)
    candidates = list(range(len(available_customers))) + [None]
    for cand in candidates:
        ttt = simulate_TTT(active_idx, truck_positions, depot_position, available_customers, cand)
        if ttt < best_ttt:
            best_ttt = ttt
            best_candidate = cand
    return best_candidate