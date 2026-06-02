import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    best_routes = None
    best_max_dist = float('inf')
    
    # Construction: regret-2 minimizing max route distance
    def construct_routes():
        routes = [[0, 0] for _ in range(truck_count)]
        route_dist = [0.0 for _ in range(truck_count)]
        for r in range(truck_count):
            route_dist[r] = distance_matrix[0, 0] * 2  # zero
        
        def compute_route_distance(route):
            d = 0.0
            for i in range(len(route)-1):
                d += distance_matrix[route[i], route[i+1]]
            return d
        
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
            for c in sorted(unassigned):  # deterministic order
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
    
    def vnd(routes, route_dist):
        max_iter = n * truck_count * 10
        vnd_iter = 0
        improved = True
        while improved and vnd_iter < max_iter:
            improved = False
            current_max = max(route_dist)
            # Neighbor 1: intra-route 2-opt
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = 0.0
                        for k in range(len(new_route)-1):
                            new_dist += distance_matrix[new_route[k], new_route[k+1]]
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
                                route_dist[r1] = sum(distance_matrix[new_route1[k], new_route1[k+1]] for k in range(len(new_route1)-1))
                                route_dist[r2] = sum(distance_matrix[new_route2[k], new_route2[k+1]] for k in range(len(new_route2)-1))
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
                            new_dist1 = sum(distance_matrix[new_route1[k], new_route1[k+1]] for k in range(len(new_route1)-1))
                            new_dist2 = sum(distance_matrix[new_route2[k], new_route2[k+1]] for k in range(len(new_route2)-1))
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
                            new_dist1 = sum(distance_matrix[new1[k], new1[k+1]] for k in range(len(new1)-1))
                            new_dist2 = sum(distance_matrix[new2[k], new2[k+1]] for k in range(len(new2)-1))
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
    
    # Initial construction and VND
    routes, route_dist = construct_routes()
    report_best_vrp(routes)
    routes, route_dist = vnd(routes, route_dist)
    best_routes = [list(r) for r in routes]
    best_max_dist = max(route_dist)
    
    # Restart phase
    max_restarts = 3
    for restart in range(max_restarts):
        # Deterministic perturbation: remove farthest customers from worst routes
        # Identify worst route(s) by max distance, tie by index
        max_dist = max(route_dist)
        worst_indices = [i for i, d in enumerate(route_dist) if d == max_dist]
        # From each worst route, remove the customer farthest from depot (tie by index)
        removed_customers = []
        for idx in worst_indices:
            route = routes[idx]
            if len(route) <= 2:
                continue
            # Find farthest from depot among interior customers
            farthest = -1
            farthest_idx = -1
            for i in range(1, len(route)-1):
                cust = route[i]
                dist = distance_matrix[0, cust]
                if dist > farthest or (dist == farthest and cust < farthest_idx):
                    farthest = dist
                    farthest_idx = i
            cust = route[farthest_idx]
            removed_customers.append(cust)
            del route[farthest_idx]
            # Recompute route distance
            new_dist = 0.0
            for k in range(len(route)-1):
                new_dist += distance_matrix[route[k], route[k+1]]
            route_dist[idx] = new_dist
        # Limit number of removed customers to 3
        if len(removed_customers) > 3:
            removed_customers = removed_customers[:3]
        if not removed_customers:
            break
        # Reinsert removed customers using regret-2 (same logic as construction but on existing routes)
        unassigned = set(removed_customers)
        while unassigned:
            best_regret = -1.0
            best_cust = None
            best_pos = None
            for c in sorted(unassigned):
                best_val = float('inf')
                second_val = float('inf')
                best_pos_c = None
                for r_idx, route in enumerate(routes):
                    curr_dist = route_dist[r_idx]
                    for i in range(1, len(route)):
                        new_dist = curr_dist - distance_matrix[route[i-1], route[i]] + distance_matrix[route[i-1], c] + distance_matrix[c, route[i]]
                        other_max = max(route_dist[:r_idx] + route_dist[r_idx+1:], default=0.0)
                        cand_max = max(new_dist, other_max)
                        if cand_max < best_val:
                            second_val = best_val
                            best_val = cand_max
                            best_pos_c = (r_idx, i)
                        elif cand_max < second_val and cand_max != best_val:
                            second_val = cand_max
                regret = second_val - best_val if second_val != float('inf') else 0.0
                if regret > best_regret or (regret == best_regret and (best_cust is None or c < best_cust)):
                    best_regret = regret
                    best_cust = c
                    best_pos = best_pos_c
            r_idx, i = best_pos
            route = routes[r_idx]
            route.insert(i, best_cust)
            route_dist[r_idx] = sum(distance_matrix[route[k], route[k+1]] for k in range(len(route)-1))
            unassigned.remove(best_cust)
        # Apply VND again
        routes, route_dist = vnd(routes, route_dist)
        current_max = max(route_dist)
        if current_max < best_max_dist - 1e-9:
            best_max_dist = current_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(routes)
    # Return best routes found
    return best_routes