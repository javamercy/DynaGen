import numpy as np
import heapq

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.ndim == 0 or available_customers.shape[0] == 0:
        return None

    # Identify active truck by closest position
    dists = np.linalg.norm(truck_positions - current_position, axis=1)
    active_idx = np.argmin(dists)

    def simulate_simple(active_idx, truck_positions, depot, customers, active_assignment):
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

        for t in range(n_trucks):
            if truck_dest[t] is None and unserved:
                scores = [np.linalg.norm(truck_pos[t] - c) + np.linalg.norm(c - depot) for c in unserved]
                nearest_idx = np.argmin(scores)
                dest = unserved.pop(nearest_idx)
                truck_dest[t] = dest
                dist = np.linalg.norm(truck_pos[t] - dest)
                truck_arrival[t] = current_time + dist
                heapq.heappush(event_queue, (truck_arrival[t], t))

        while event_queue or unserved:
            if not event_queue and unserved:
                for t in range(n_trucks):
                    if truck_dest[t] is None and unserved:
                        scores = [np.linalg.norm(truck_pos[t] - c) + np.linalg.norm(c - depot) for c in unserved]
                        nearest_idx = np.argmin(scores)
                        dest = unserved.pop(nearest_idx)
                        truck_dest[t] = dest
                        dist = np.linalg.norm(truck_pos[t] - dest)
                        truck_arrival[t] = current_time + dist
                        heapq.heappush(event_queue, (truck_arrival[t], t))
            time, t_idx = heapq.heappop(event_queue)
            current_time = time
            truck_pos[t_idx] = truck_dest[t_idx].copy()
            last_service_time[t_idx] = current_time
            truck_dest[t_idx] = None
            if unserved:
                scores = [np.linalg.norm(truck_pos[t_idx] - c) + np.linalg.norm(c - depot) for c in unserved]
                nearest_idx = np.argmin(scores)
                dest = unserved.pop(nearest_idx)
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

    best_ttt = float('inf')
    best_candidate = None
    for i in range(len(available_customers)):
        ttt = simulate_simple(active_idx, truck_positions, depot_position, available_customers, i)
        if ttt < best_ttt:
            best_ttt = ttt
            best_candidate = i
    ttt_wait = simulate_simple(active_idx, truck_positions, depot_position, available_customers, None)
    if ttt_wait < best_ttt:
        best_candidate = None
    return best_candidate