import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    # Helper to compute route length
    def route_length(route):
        if len(route) <= 2:
            return 0.0
        total = 0.0
        for i in range(len(route) - 1):
            total += distance_matrix[route[i], route[i+1]]
        return total
    
    # Initialize empty routes
    routes = [[0, 0] for _ in range(truck_count)]
    
    # --- Regret insertion construction (from cand_000007) ---
    unassigned = set(customers)
    while unassigned:
        best_cust = None
        best_regret = -1e9
        best_cost = None
        best_route_idx = None
        best_pos = None
        for cust in unassigned:
            insert_costs = []
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    cost = (distance_matrix[prev, cust] + 
                            distance_matrix[cust, nxt] - 
                            distance_matrix[prev, nxt])
                    insert_costs.append((cost, r_idx, pos))
            if not insert_costs:
                continue
            insert_costs.sort(key=lambda x: x[0])
            best = insert_costs[0][0]
            second = insert_costs[1][0] if len(insert_costs) > 1 else best + 1e9
            regret = second - best
            # Tie-break: larger regret, then larger best cost (to discourage long routes), then smaller customer index
            if (regret > best_regret or 
                (abs(regret - best_regret) < 1e-12 and best_cost is not None and best > best_cost) or
                (abs(regret - best_regret) < 1e-12 and best_cost is not None and abs(best - best_cost) < 1e-12 and cust < best_cust)):
                best_regret = regret
                best_cost = best
                best_cust = cust
                best_route_idx = insert_costs[0][1]
                best_pos = insert_costs[0][2]
        if best_cust is None:
            break
        route = routes[best_route_idx]
        route.insert(best_pos, best_cust)
        unassigned.remove(best_cust)
    
    # Initial improvement report
    current_max = max(route_length(r) for r in routes)
    report_best_vrp(routes)
    
    # --- Improvement phase: VND with relocate, swap, 2-opt ---
    max_iter = max(n, 10)  # bound iterations
    for _ in range(max_iter):
        improved = False
        
        # 1) Inter-route relocate: move a customer from the longest route to another route to reduce max
        lengths = [route_length(r) for r in routes]
        max_idx = max(range(truck_count), key=lambda i: lengths[i])
        max_route = routes[max_idx]
        if len(max_route) > 3:  # at least one customer
            # iterate over customers in longest route (excluding depots)
            for cust_idx, cust in enumerate(max_route[1:-1]):
                # create candidate route without this customer
                new_max_route = [0] + [c for c in max_route[1:-1] if c != cust] + [0]
                new_max_len = route_length(new_max_route)
                # try inserting this customer into another route
                for r_idx in range(truck_count):
                    if r_idx == max_idx:
                        continue
                    other_route = routes[r_idx]
                    for pos in range(1, len(other_route)):
                        new_other = other_route[:pos] + [cust] + other_route[pos:]
                        new_other_len = route_length(new_other)
                        # compute new max
                        new_max_candidate = new_max_len
                        for rr in range(truck_count):
                            if rr == max_idx:
                                cand_len = new_max_len
                            elif rr == r_idx:
                                cand_len = new_other_len
                            else:
                                cand_len = lengths[rr]
                            if cand_len > new_max_candidate:
                                new_max_candidate = cand_len
                        if new_max_candidate < current_max - 1e-12:
                            # Apply move
                            routes[max_idx] = new_max_route
                            routes[r_idx] = new_other
                            current_max = new_max_candidate
                            report_best_vrp(routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        if improved:
            continue
        
        # 2) Inter-route swap: exchange customers between two routes
        for i in range(truck_count):
            for j in range(i+1, truck_count):
                route_i = routes[i]
                route_j = routes[j]
                if len(route_i) <= 2 or len(route_j) <= 2:
                    continue
                # get customers (excluding depots)
                custs_i = route_i[1:-1]
                custs_j = route_j[1:-1]
                for cust_i in custs_i:
                    for cust_j in custs_j:
                        # create new routes by swapping the customers
                        # we need to keep order of other customers, so just replace the first occurrence? 
                        # Instead, we'll remove both and insert at positions that minimize increase? 
                        # For simplicity, we'll just replace the customer at the same index? That might not be valid if positions differ.
                        # We'll use a more general approach: remove cust_i from i and cust_j from j, then insert cust_i into j at best position and cust_j into i at best position.
                        # But to keep it efficient, we'll compute the new routes by swapping the customers directly (replace cust_i with cust_j in route_i and vice versa).
                        new_i = [0] + [c if c != cust_i else cust_j for c in route_i[1:-1]] + [0]
                        new_j = [0] + [c if c != cust_j else cust_i for c in route_j[1:-1]] + [0]
                        # Note: This may result in duplicate customers if cust_i == cust_j (should not happen) or if both routes contain the same customer (impossible).
                        # However, this simple swap might change the sequence in a non-optimal way. But it's a valid swap.
                        # To avoid issues, we'll compute proper insertion after removal:
                        # Actually we need to ensure the routes are permutations. Better: remove cust_i from i, remove cust_j from j, then insert cust_i into j and cust_j into i at best positions.
                        # We'll implement that.
                        # But for brevity, we'll use the simple swap (replace) since it's a valid move.
                        # Compute lengths
                        len_i_new = route_length(new_i)
                        len_j_new = route_length(new_j)
                        new_max_candidate = max(len_i_new, len_j_new)
                        for rr in range(truck_count):
                            if rr not in (i, j):
                                new_max_candidate = max(new_max_candidate, lengths[rr])
                        if new_max_candidate < current_max - 1e-12:
                            routes[i] = new_i
                            routes[j] = new_j
                            current_max = new_max_candidate
                            report_best_vrp(routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        
        # 3) Intra-route 2-opt on each route
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 3:
                continue
            best_route = route[:]
            best_len = route_length(route)
            for i in range(1, len(route)-2):
                for k in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                    new_len = route_length(new_route)
                    if new_len < best_len - 1e-12:
                        best_route = new_route
                        best_len = new_len
            if best_len < route_length(route) - 1e-12:
                routes[r_idx] = best_route
                new_max = max(route_length(r) for r in routes)
                if new_max < current_max - 1e-12:
                    current_max = new_max
                    report_best_vrp(routes)
                improved = True
        if not improved:
            break
    
    return routes