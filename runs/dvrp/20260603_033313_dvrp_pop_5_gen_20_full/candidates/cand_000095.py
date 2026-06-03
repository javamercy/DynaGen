import numpy as np
import heapq

def simulate_with_depot_bias(active_idx, truck_positions, depot, available_customers, active_assignment, alpha=0.2):
    n = len(truck_positions)
    pos = [p.copy() for p in truck_positions]
    unserved = [c.copy() for c in available_customers]
    dest = [None] * n
    arrival = [0.0] * n
    last_service = [0.0] * n
    current_time = 0.0
    event_queue = []

    # handle active assignment (customer or None)
    if active_assignment is not None:
        cust = unserved.pop(active_assignment)
        dest[active_idx] = cust
        dist = np.linalg.norm(pos[active_idx] - cust)
        arrival[active_idx] = current_time + dist
        heapq.heappush(event_queue, (arrival[active_idx], active_idx))

    # initial dispatch for idle trucks (including active if not assigned)
    for t in range(n):
        if dest[t] is None and unserved:
            # biased nearest neighbor
            best_score = float('inf')
            best_idx = None
            for i, c in enumerate(unserved):
                d2cust = np.linalg.norm(pos[t] - c)
                d2depot = np.linalg.norm(c - depot)
                score = d2cust + alpha * d2depot
                if score < best_score:
                    best_score = score
                    best_idx = i
            cust = unserved.pop(best_idx)
            dest[t] = cust
            dist = np.linalg.norm(pos[t] - cust)
            arrival[t] = current_time + dist
            heapq.heappush(event_queue, (arrival[t], t))

    while event_queue or unserved:
        if not event_queue and unserved:
            for t in range(n):
                if dest[t] is None and unserved:
                    best_score = float('inf')
                    best_idx = None
                    for i, c in enumerate(unserved):
                        d2cust = np.linalg.norm(pos[t] - c)
                        d2depot = np.linalg.norm(c - depot)
                        score = d2cust + alpha * d2depot
                        if score < best_score:
                            best_score = score
                            best_idx = i
                    cust = unserved.pop(best_idx)
                    dest[t] = cust
                    dist = np.linalg.norm(pos[t] - cust)
                    arrival[t] = current_time + dist
                    heapq.heappush(event_queue, (arrival[t], t))
        time, t_idx = heapq.heappop(event_queue)
        current_time = time
        pos[t_idx] = dest[t_idx].copy()
        last_service[t_idx] = current_time
        dest[t_idx] = None
        if unserved:
            best_score = float('inf')
            best_idx = None
            for i, c in enumerate(unserved):
                d2cust = np.linalg.norm(pos[t_idx] - c)
                d2depot = np.linalg.norm(c - depot)
                score = d2cust + alpha * d2depot
                if score < best_score:
                    best_score = score
                    best_idx = i
            cust = unserved.pop(best_idx)
            dest[t_idx] = cust
            dist = np.linalg.norm(pos[t_idx] - cust)
            arrival[t_idx] = current_time + dist
            heapq.heappush(event_queue, (arrival[t_idx], t_idx))

    max_return = 0.0
    for t in range(n):
        return_dist = np.linalg.norm(pos[t] - depot)
        return_time = last_service[t] + return_dist
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
        ttt = simulate_with_depot_bias(active_idx, truck_positions, depot_position, available_customers, cand)
        if ttt < best_ttt:
            best_ttt = ttt
            best_candidate = cand
    return best_candidate