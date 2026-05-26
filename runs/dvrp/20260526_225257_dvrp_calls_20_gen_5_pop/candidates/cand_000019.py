import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    n_available = len(available_customers)
    if n_available == 0:
        return None
    
    n_trucks = truck_positions.shape[0]
    # identify deciding truck index
    dist_to_current = np.linalg.norm(truck_positions - current_position, axis=1)
    deciding_truck_idx = np.argmin(dist_to_current)
    
    # distances from each available customer to depot
    dist_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    
    # compute regrets
    regrets = np.empty(n_available)
    for i, cust in enumerate(available_customers):
        d_curr = np.linalg.norm(current_position - cust)
        cost_now = current_time + d_curr + dist_to_depot[i]
        best_other = float('inf')
        for j in range(n_trucks):
            if j == deciding_truck_idx:
                continue
            d_truck = np.linalg.norm(truck_positions[j] - cust)
            cost_other = current_time + d_truck + dist_to_depot[i]
            if cost_other < best_other:
                best_other = cost_other
        if n_trucks == 1:
            regrets[i] = -float('inf')
        else:
            regrets[i] = cost_now - best_other
    
    # filter non-positive regret candidates
    candidates = [i for i in range(n_available) if regrets[i] <= 0]
    if not candidates:
        return None
    
    # helper to compute route time
    def route_time(waypoints):
        t = 0.0
        for p in range(len(waypoints)-1):
            t += np.linalg.norm(waypoints[p] - waypoints[p+1])
        return t
    
    # greedy insertion: given starting routes (list of lists) and list of customers, return max route time
    def greedy_insert(start_routes, customers, allow_trucks=None):
        routes = [list(seq) for seq in start_routes]  # deep copy
        for cust in customers:
            best_truck = -1
            best_pos = -1
            best_inc = float('inf')
            for t_idx in range(n_trucks):
                if allow_trucks is not None and t_idx not in allow_trucks:
                    continue
                waypoints = routes[t_idx]
                for p in range(len(waypoints)-1):
                    prev = waypoints[p]
                    nxt = waypoints[p+1]
                    inc = np.linalg.norm(prev - cust) + np.linalg.norm(cust - nxt) - np.linalg.norm(prev - nxt)
                    if inc < best_inc:
                        best_inc = inc
                        best_truck = t_idx
                        best_pos = p+1
            routes[best_truck].insert(best_pos, cust)
        max_route = 0.0
        for waypoints in routes:
            rt = route_time(waypoints)
            if rt > max_route:
                max_route = rt
        return max_route
    
    # lookahead size
    L = min(10, n_available)
    # indices of farthest customers from depot
    sorted_by_dist = np.argsort(dist_to_depot)[::-1]
    high_impact_indices = sorted_by_dist[:L].tolist()
    
    # waiting evaluation: insert high impact customers into all trucks except deciding truck
    start_routes_wait = []
    for t in range(n_trucks):
        start_routes_wait.append([truck_positions[t].copy(), depot_position.copy()])
    waiting_customers = [available_customers[i].copy() for i in high_impact_indices]
    waiting_ttt = current_time + greedy_insert(start_routes_wait, waiting_customers,
                                               allow_trucks=[t for t in range(n_trucks) if t != deciding_truck_idx])
    
    best_candidate = None
    best_candidate_ttt = float('inf')
    for cand_idx in candidates:
        # insertion set: top L customers excluding candidate
        insertion_set = []
        for idx in sorted_by_dist:
            if len(insertion_set) >= L:
                break
            if idx != cand_idx:
                insertion_set.append(idx)
        # build start routes: candidate first for deciding truck
        start_routes_cand = []
        for t in range(n_trucks):
            if t == deciding_truck_idx:
                start_routes_cand.append([truck_positions[t].copy(), available_customers[cand_idx].copy(), depot_position.copy()])
            else:
                start_routes_cand.append([truck_positions[t].copy(), depot_position.copy()])
        customers_to_insert = [available_customers[i].copy() for i in insertion_set]
        cand_ttt = current_time + greedy_insert(start_routes_cand, customers_to_insert, allow_trucks=None)
        if cand_ttt < best_candidate_ttt:
            best_candidate_ttt = cand_ttt
            best_candidate = cand_idx
    
    if waiting_ttt < best_candidate_ttt:
        return None
    else:
        return best_candidate