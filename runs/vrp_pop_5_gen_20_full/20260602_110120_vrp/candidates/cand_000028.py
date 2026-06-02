import numpy as np
from itertools import combinations

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    
    def compute_route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    # Build initial solution using regret-2 insertion
    def build_initial():
        routes = [[0, 0] for _ in range(truck_count)]
        route_dist = [0.0 for _ in range(truck_count)]
        unassigned = set(range(1, n))
        
        def best_max_and_second(customer):
            best_val = float('inf')
            best_pos = None
            second_val = float('inf')
            for r_idx, route in enumerate(routes):
                curr_dist = route_dist[r_idx]
                for i in range(1, len(route)):
                    new_dist = curr_dist - distance_matrix[route[i-1], route[i]] + distance_matrix[route[i-1], customer] + distance_matrix[customer, route[i]]
                    other_max = max(route_dist[:r_idx] + route_dist[r_idx+1:], default=0.0)
                    cand_max = max(new_dist, other_max)
                    if cand_max < best_val:
                        second_val = best_val
                        best_val = cand_max
                        best_pos = (r_idx, i)
                    elif cand_max < second_val and cand_max != best_val:
                        second_val = cand_max
            return best_val, second_val, best_pos
        
        while unassigned:
            best_regret = -1.0
            best_customer = None
            best_insertion = None
            for c in unassigned:
                best_val, second_val, best_pos = best_max_and_second(c)
                regret = second_val - best_val if second_val != float('inf') else 0.0
                if regret > best_regret or (regret == best_regret and (best_customer is None or c < best_customer)):
                    best_regret = regret
                    best_customer = c
                    best_insertion = best_pos
            r_idx, i = best_insertion
            route = routes[r_idx]
            route.insert(i, best_customer)
            route_dist[r_idx] = compute_route_distance(route)
            unassigned.remove(best_customer)
        return routes, route_dist
    
    # VND
    def vnd(routes, route_dist):
        max_iter = n * truck_count
        vnd_iter = 0
        improved = True
        current_max = max(route_dist)
        while improved and vnd_iter < max_iter:
            improved = False
            # Neighbor 1: intra-route 2-opt
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = compute_route_distance(new_route)
                        if len(new_route) != len(route):
                            continue
                        if new_dist < route_dist[r_idx]:
                            new_max = max(new_dist, max(route_dist[:r_idx] + route_dist[r_idx+1:], default=0.0))
                            if new_max < current_max - 1e-9:
                                routes[r_idx] = new_route
                                route_dist[r_idx] = new_dist
                                current_max = new_max
                                improved = True
                                report_best_vrp(routes)
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                vnd_iter += 1
                continue
            # Neighbor 2: inter-route relocate
            for r1 in range(truck_count):
                route1 = routes[r1]
                if len(route1) <= 2:
                    continue
                for i in range(1, len(route1)-1):
                    cust = route1[i]
                    for r2 in range(truck_count):
                        if r2 == r1:
                            continue
                        route2 = routes[r2]
                        for pos in range(1, len(route2)):
                            prev_rem = route1[i-1]
                            next_rem = route1[i+1]
                            removed_cost = distance_matrix[prev_rem, cust] + distance_matrix[cust, next_rem] - distance_matrix[prev_rem, next_rem]
                            new_dist_r1 = route_dist[r1] - removed_cost
                            prev_ins = route2[pos-1]
                            next_ins = route2[pos]
                            added_cost = distance_matrix[prev_ins, cust] + distance_matrix[cust, next_ins] - distance_matrix[prev_ins, next_ins]
                            new_dist_r2 = route_dist[r2] + added_cost
                            other_max = max(route_dist[:r1] + route_dist[r1+1:r2] + route_dist[r2+1:], default=0.0)
                            new_max = max(new_dist_r1, new_dist_r2, other_max)
                            if new_max < current_max - 1e-9:
                                new_route1 = route1[:i] + route1[i+1:]
                                if len(new_route1) == 2:
                                    new_route1 = [0, 0]
                                new_route2 = route2[:pos] + [cust] + route2[pos:]
                                routes[r1] = new_route1
                                routes[r2] = new_route2
                                route_dist[r1] = compute_route_distance(new_route1)
                                route_dist[r2] = compute_route_distance(new_route2)
                                current_max = new_max
                                improved = True
                                report_best_vrp(routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                vnd_iter += 1
                continue
            # Neighbor 3: inter-route swap
            for r1 in range(truck_count):
                route1 = routes[r1]
                for r2 in range(r1+1, truck_count):
                    route2 = routes[r2]
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            cust1 = route1[i]
                            cust2 = route2[j]
                            new_route1 = route1[:i] + [cust2] + route1[i+1:]
                            new_route2 = route2[:j] + [cust1] + route2[j+1:]
                            new_dist1 = compute_route_distance(new_route1)
                            new_dist2 = compute_route_distance(new_route2)
                            other_max = max(route_dist[:r1] + route_dist[r1+1:r2] + route_dist[r2+1:], default=0.0)
                            new_max = max(new_dist1, new_dist2, other_max)
                            if new_max < current_max - 1e-9:
                                routes[r1] = new_route1
                                routes[r2] = new_route2
                                route_dist[r1] = new_dist1
                                route_dist[r2] = new_dist2
                                current_max = new_max
                                improved = True
                                report_best_vrp(routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                vnd_iter += 1
                continue
            # Neighbor 4: inter-route 2-opt* (cross)
            for r1 in range(truck_count):
                route1 = routes[r1]
                for r2 in range(r1+1, truck_count):
                    route2 = routes[r2]
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new1 = route1[:i] + route2[j:]
                            new2 = route2[:j] + route1[i:]
                            new_dist1 = compute_route_distance(new1)
                            new_dist2 = compute_route_distance(new2)
                            other_max = max(route_dist[:r1] + route_dist[r1+1:r2] + route_dist[r2+1:], default=0.0)
                            new_max = max(new_dist1, new_dist2, other_max)
                            if new_max < current_max - 1e-9:
                                routes[r1] = new1
                                routes[r2] = new2
                                route_dist[r1] = new_dist1
                                route_dist[r2] = new_dist2
                                current_max = new_max
                                improved = True
                                report_best_vrp(routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            vnd_iter += 1
        return routes, route_dist
    
    # Initial construction
    routes, route_dist = build_initial()
    report_best_vrp(routes)
    
    # VND on initial
    routes, route_dist = vnd(routes, route_dist)
    
    # Best solution tracking
    best_routes = [list(r) for r in routes]
    best_max = max(route_dist)
    
    # Deterministic restarts with new perturbation: relocate best saving customer from max route
    max_restarts = 3
    for restart in range(max_restarts):
        perturbed_routes = [list(r) for r in best_routes]
        # Identify route with max distance
        max_dist = max(route_dist)
        candidate_routes = [r for r, d in enumerate(route_dist) if abs(d - max_dist) < 1e-9]
        # deterministic tie: smallest route index
        max_route_idx = min(candidate_routes)
        route_max = perturbed_routes[max_route_idx]
        if len(route_max) <= 2:
            continue
        # Find customer in that route whose removal yields largest reduction (best saving)
        best_saving = -1.0
        best_cust = None
        best_i = None
        for i in range(1, len(route_max)-1):
            cust = route_max[i]
            prev = route_max[i-1]
            next = route_max[i+1]
            removed_cost = distance_matrix[prev, cust] + distance_matrix[cust, next] - distance_matrix[prev, next]
            saving = removed_cost  # positive reduction
            if saving > best_saving or (abs(saving - best_saving) < 1e-9 and (best_cust is None or cust < best_cust)):
                best_saving = saving
                best_cust = cust
                best_i = i
        if best_cust is None:
            continue
        # Remove best_cust from max route
        del perturbed_routes[max_route_idx][best_i]
        # Find best route to insert (minimizing new max distance)
        current_dist_without = compute_route_distance(perturbed_routes[max_route_idx])
        old_max = max_dist  # note: the max route distance might have changed after removal
        best_new_max = float('inf')
        best_r_insert = None
        best_pos_insert = None
        for r_idx in range(truck_count):
            route = perturbed_routes[r_idx]
            for pos in range(1, len(route)):
                new_route = route[:pos] + [best_cust] + route[pos:]
                new_dist = compute_route_distance(new_route)
                other_distances = [compute_route_distance(perturbed_routes[k]) if k != r_idx else new_dist for k in range(truck_count)]
                cand_max = max(other_distances)
                if cand_max < best_new_max - 1e-9 or (abs(cand_max - best_new_max) < 1e-9 and (best_r_insert is None or r_idx < best_r_insert or (r_idx == best_r_insert and pos < best_pos_insert))):
                    best_new_max = cand_max
                    best_r_insert = r_idx
                    best_pos_insert = pos
        if best_r_insert is not None:
            perturbed_routes[best_r_insert].insert(best_pos_insert, best_cust)
        else:
            # reinsert at original position? fallback: just reinsert at same route? but we must maintain feasibility
            # For safety, revert to best_routes
            perturbed_routes = [list(r) for r in best_routes]
        # Compute distances for perturbed
        perturbed_dist = [compute_route_distance(route) for route in perturbed_routes]
        # Run VND on perturbed
        perturbed_routes, perturbed_dist = vnd(perturbed_routes, perturbed_dist)
        perturbed_max = max(perturbed_dist)
        if perturbed_max < best_max - 1e-9:
            best_routes = [list(r) for r in perturbed_routes]
            best_max = perturbed_max
            report_best_vrp(best_routes)
    
    return best_routes