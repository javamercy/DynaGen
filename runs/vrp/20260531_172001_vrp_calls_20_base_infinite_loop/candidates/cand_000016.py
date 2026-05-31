import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]
    
    def route_length(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total
    
    def max_route_len(routes):
        return max(route_length(r) for r in routes)
    
    best_routes = None
    best_max = float('inf')
    
    # Multi-start: run construction a few times (n//10, but at least 1)
    max_attempts = max(1, n // 10)
    for attempt in range(max_attempts):
        # Construction: hybrid regret-minmax
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = list(range(1, n))
        while unassigned:
            # For each unassigned customer, compute best insertion data
            candidate_info = []
            for cust in unassigned:
                # compute all insertion costs for this customer
                costs = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        costs.append((cost, r_idx, pos))
                # sort by cost
                costs.sort(key=lambda x: x[0])
                best_cost = costs[0][0]
                best_r_idx = costs[0][1]
                best_pos = costs[0][2]
                second_cost = costs[1][0] if len(costs) > 1 else best_cost + 1e9
                regret = second_cost - best_cost
                # compute resulting max if this customer inserted in best position
                # compute new length for that route
                old_len = route_length(routes[best_r_idx])
                new_len = old_len + best_cost
                other_lens = [route_length(r) for i,r in enumerate(routes) if i != best_r_idx]
                new_max = max(new_len, *other_lens)
                candidate_info.append((cust, regret, new_max, best_r_idx, best_pos))
            # Sort candidates by regret descending, then by new_max ascending
            candidate_info.sort(key=lambda x: (-x[1], x[2]))
            # Consider top 3 regret (or all if less)
            k = min(3, len(candidate_info))
            # Among top k, choose the one with smallest new_max
            best_candidate = min(candidate_info[:k], key=lambda x: x[2])
            cust, _, _, r_idx, pos = best_candidate
            routes[r_idx].insert(pos, cust)
            unassigned.remove(cust)
        
        current_max = max_route_len(routes)
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)
        
        # Improvement loop
        max_iter = n * truck_count
        iter_count = 0
        no_improve_count = 0
        perturb_threshold = max(1, n // 3)
        improved = True
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            lengths = [route_length(r) for r in routes]
            current_max = max(lengths)
            
            # 1. Inter-route relocate from longest route
            max_idx = np.argmax(lengths)
            max_route = routes[max_idx]
            if len(max_route) > 2:
                best_delta = 0
                best_move = None
                for cust in max_route[1:-1]:
                    new_max_route = [x for x in max_route if x != cust]
                    new_max_len = route_length(new_max_route)
                    for r_idx in range(truck_count):
                        if r_idx == max_idx:
                            continue
                        other_route = routes[r_idx]
                        for pos in range(1, len(other_route)):
                            new_other = other_route[:pos] + [cust] + other_route[pos:]
                            new_other_len = route_length(new_other)
                            new_max_candidate = max(new_max_len, new_other_len, *[lengths[i] for i in range(truck_count) if i not in (max_idx, r_idx)])
                            if new_max_candidate < current_max - 1e-12:
                                delta = current_max - new_max_candidate
                                if delta > best_delta:
                                    best_delta = delta
                                    best_move = (cust, max_idx, r_idx, pos)
                if best_move:
                    cust, from_idx, to_idx, pos = best_move
                    routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                    routes[to_idx].insert(pos, cust)
                    current_max -= best_delta
                    improved = True
                    if current_max < best_max - 1e-12:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(routes)
            
            if improved:
                continue
            
            # 2. Inter-route swap (simplified: try swapping customers between two routes)
            swap_improved = False
            for i in range(truck_count):
                for j in range(i+1, truck_count):
                    route_i = routes[i]
                    route_j = routes[j]
                    if len(route_i) <= 2 or len(route_j) <= 2:
                        continue
                    for cust_i in route_i[1:-1]:
                        for cust_j in route_j[1:-1]:
                            # Remove both and reinsert at best positions? For simplicity, swap positions
                            # We'll compute new routes by removing cust_i from i and cust_j from j,
                            # then inserting each into the other route at original positions (heuristic)
                            # Actually, we can just swap the customers: remove both, then add cust_j to i at the spot of cust_i, and cust_i to j at spot of cust_j
                            # But positions may change; we'll keep order of other customers
                            temp_i = [c for c in route_i[1:-1] if c != cust_i]
                            temp_j = [c for c in route_j[1:-1] if c != cust_j]
                            # Insert cust_j into temp_i at some position (e.g., at original index of cust_i if still valid)
                            # To keep things simple, we'll insert at a position that minimizes the length of that route
                            # But that makes it complex; we'll just insert at the end of the sequence (near depot).
                            # However, we need the route to end at depot, so insertion positions after last customer.
                            # Better: we'll compute length directly by constructing routes as lists and then inserting at best position?
                            # Given time, we'll skip this neighborhood to keep code manageable; rely on relocate and 2-opt.
                            pass
            
            # 3. Intra-route 2-opt on each route
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for k in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                        if route_length(new_route) < route_length(route) - 1e-12:
                            routes[r_idx] = new_route
                            improved = True
                            new_max = max_route_len(routes)
                            if new_max < current_max - 1e-12:
                                current_max = new_max
                                if current_max < best_max - 1e-12:
                                    best_max = current_max
                                    best_routes = [r[:] for r in routes]
                                    report_best_vrp(routes)
                            break
                    if improved:
                        break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count >= perturb_threshold:
                    # Perturbation: move ~20% of customers randomly
                    customers = list(range(1, n))
                    random.shuffle(customers)
                    num_perturb = max(1, n // 5)
                    for cust in customers[:num_perturb]:
                        # remove from current route
                        for r_idx, route in enumerate(routes):
                            if cust in route:
                                routes[r_idx] = [x for x in route if x != cust]
                                break
                        # insert into random route at random position
                        r_idx = random.randrange(truck_count)
                        pos = random.randrange(1, len(routes[r_idx]))
                        routes[r_idx].insert(pos, cust)
                    current_max = max_route_len(routes)
                    no_improve_count = 0
                    improved = True
        # End of improvement for this attempt
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)
    
    if best_routes is None:
        best_routes = routes
    return best_routes