import numpy as np
import heapq

def simulate_TTT_depot_penalty_refined(active_idx, truck_positions, depot, customers, active_assignment):
    n_trucks = len(truck_positions)
    truck_pos = [truck_positions[i].copy() for i in range(n_trucks)]
    unserved = [c.copy() for c in customers]
    truck_dest = [None] * n_trucks
    truck_arrival = [None] * n_trucks
    last_service_time = [0.0] * n_trucks
    current_time = 0.0
    event_queue = []

    if active_assignment is not None:
        dest = unserved.pop(active_assignment)
        truck_dest[active_idx] = dest
        dist = np.linalg.norm(truck_pos[active_idx] - dest)
        truck_arrival[active_idx] = current_time + dist
        heapq.heappush(event_queue, (truck_arrival[active_idx], active_idx))

    def choose_customer(truck_idx, customers_list):
        if not customers_list:
            return None
        truck_loc = truck_pos[truck_idx]
        # Modified scoring: increased penalty coefficient from 0.1 to 0.2
        dist_truck_depot = np.linalg.norm(truck_loc - depot)
        penalty_factor = 1.0 + 0.2 * dist_truck_depot
        scores = [np.linalg.norm(truck_loc - c) + penalty_factor * np.linalg.norm(c - depot) for c in customers_list]
        return np.argmin(scores)

    for t in range(n_trucks):
        if truck_dest[t] is None and unserved:
            idx = choose_customer(t, unserved)
            if idx is not None:
                dest = unserved.pop(idx)
                truck_dest[t] = dest
                dist = np.linalg.norm(truck_pos[t] - dest)
                truck_arrival[t] = current_time + dist
                heapq.heappush(event_queue, (truck_arrival[t], t))

    while event_queue or unserved:
        if not event_queue and unserved:
            for t in range(n_trucks):
                if truck_dest[t] is None and unserved:
                    idx = choose_customer(t, unserved)
                    if idx is not None:
                        dest = unserved.pop(idx)
                        truck_dest[t] = dest
                        dist = np.linalg.norm(truck_pos[t] - dest)
                        truck_arrival[t] = current_time + dist
                        heapq.heappush(event_queue, (truck_arrival[t], t))
            continue
        time, t_idx = heapq.heappop(event_queue)
        current_time = time
        truck_pos[t_idx] = truck_dest[t_idx].copy()
        last_service_time[t_idx] = current_time
        truck_dest[t_idx] = None
        if unserved:
            idx = choose_customer(t_idx, unserved)
            if idx is not None:
                dest = unserved.pop(idx)
                truck_dest[t_idx] = dest
                dist = np.linalg.norm(truck_pos[t_idx] - dest)
                truck_arrival[t_idx] = current_time + dist
                heapq.heappush(event_queue, (truck_arrival[t_idx], t_idx))

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
    active_idx = np.where((truck_positions == current_position).all(axis=1))[0]
    if len(active_idx) == 0:
        dists = np.linalg.norm(truck_positions - current_position, axis=1)
        active_idx = np.argmin(dists)
    else:
        active_idx = active_idx[0]
    best_ttt = float('inf')
    best_candidate = None
    candidates = list(range(len(available_customers))) + [None]
    for cand in candidates:
        ttt = simulate_TTT_depot_penalty_refined(active_idx, truck_positions, depot_position, available_customers, cand)
        if ttt < best_ttt:
            best_ttt = ttt
            best_candidate = cand
    return best_candidate