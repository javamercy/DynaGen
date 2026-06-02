import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    
    def compute_route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def construction():
        routes = [[0, 0] for _ in range(truck_count)]
        route_distances = [0.0 for _ in range(truck_count)]
        unassigned = set(range(1, n))
        
        def best_insertions(customer):
            # returns list of (cost, route_idx, position) sorted by cost
            options = []
            for r_idx, route in enumerate(routes):
                curr_dist = route_distances[r_idx]
                for i in range(1, len(route)):
                    new_dist = curr_dist - distance_matrix[route[i-1], route[i]] \
                               + distance_matrix[route[i-1], customer] \
                               + distance_matrix[customer, route[i]]
                    other_max = max(route_distances[:r_idx] + route_distances[r_idx+1:], default=0.0)
                    cand_max = max(new_dist, other_max)
                    options.append((cand_max, r_idx, i))
            options.sort(key=lambda x: x[0])
            return options
        
        while unassigned:
            regrets = []
            for c in unassigned:
                opts = best_insertions(c)
                # compute regret-3
                if len(opts) >= 3:
                    best_cost = opts[0][0]
                    regret = sum(opts[i][0] - best_cost for i in range(3))
                else:
                    regret = 0.0
                # tie-break by best cost (lower is better)
                best_cost = opts[0][0] if opts else float('inf')
                regrets.append((-regret, best_cost, c, opts))
            # select customer with max regret then min best_cost
            regrets.sort(key=lambda x: (x[0], x[1]))
            selected = regrets[0][2]
            opts = regrets[0][3]
            # choose best insertion for selected
            cost, r_idx, i = opts[0]
            route = routes[r_idx]
            route.insert(i, selected)
            route_distances[r_idx] = compute_route_distance(route)
            unassigned.remove(selected)
        return routes, route_distances

    def local_search(routes, route_distances):
        current_routes = [list(r) for r in routes]
        current_distances = list(route_distances)
        current_max = max(current_distances)
        improved = True
        max_iters = n * n
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            # Intra-route 2-opt
            for r_idx in range(truck_count):
                route = current_routes[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = compute_route_distance(new_route)
                        if new_dist < current_distances[r_idx] - 1e-12:
                            current_distances[r_idx] = new_dist
                            current_routes[r_idx] = new_route
                            new_max = max(current_distances)
                            if new_max < current_max - 1e-12:
                                current_max = new_max
                                report_best_vrp(current_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Inter-swap
            for r1 in range(truck_count):
                for r2 in range(r1+1, truck_count):
                    route1 = current_routes[r1]
                    route2 = current_routes[r2]
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new1 = route1[:i] + [route2[j]] + route1[i+1:]
                            new2 = route2[:j] + [route1[i]] + route2[j+1:]
                            new_dist1 = compute_route_distance(new1)
                            new_dist2 = compute_route_distance(new2)
                            other_max = max(current_distances[k] for k in range(truck_count) if k not in (r1, r2))
                            cand_max = max(new_dist1, new_dist2, other_max)
                            if cand_max < current_max - 1e-12:
                                current_routes[r1] = new1
                                current_routes[r2] = new2
                                current_distances[r1] = new_dist1
                                current_distances[r2] = new_dist2
                                current_max = cand_max
                                report_best_vrp(current_routes)
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
            # Inter-relocate
            for r1 in range(truck_count):
                for r2 in range(truck_count):
                    if r1 == r2:
                        continue
                    route1 = current_routes[r1]
                    route2 = current_routes[r2]
                    for i in range(1, len(route1)-1):
                        c = route1[i]
                        new1 = route1[:i] + route1[i+1:]
                        new_dist1 = compute_route_distance(new1)
                        best_new2 = None
                        best_new_dist2 = float('inf')
                        for j in range(1, len(route2)):
                            new2 = route2[:j] + [c] + route2[j:]
                            d2 = compute_route_distance(new2)
                            if d2 < best_new_dist2:
                                best_new_dist2 = d2
                                best_new2 = new2
                        other_max = max(current_distances[:r1] + current_distances[r1+1:r2] + current_distances[r2+1:], default=0.0)
                        cand_max = max(new_dist1, best_new_dist2, other_max)
                        if cand_max < current_max - 1e-12:
                            current_routes[r1] = new1
                            current_routes[r2] = best_new2
                            current_distances[r1] = new_dist1
                            current_distances[r2] = best_new_dist2
                            current_max = cand_max
                            report_best_vrp(current_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return current_routes, current_distances, current_max

    def perturb(routes, route_distances):
        new_routes = [list(r) for r in routes]
        new_distances = list(route_distances)
        for r_idx in range(truck_count):
            route = new_routes[r_idx]
            best_worsen = -float('inf')
            best_move = None
            best_new_dist = None
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    candidate = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    d = compute_route_distance(candidate)
                    worsen = d - new_distances[r_idx]
                    if worsen > best_worsen and worsen > 1e-12:
                        best_worsen = worsen
                        best_move = candidate
                        best_new_dist = d
            if best_move is not None:
                new_routes[r_idx] = best_move
                new_distances[r_idx] = best_new_dist
        return new_routes, new_distances

    best_routes = None
    best_max_val = float('inf')
    max_restarts = min(5, n)
    
    routes, route_distances = construction()
    routes, route_distances, current_max = local_search(routes, route_distances)
    if current_max < best_max_val - 1e-12:
        best_max_val = current_max
        best_routes = routes
        report_best_vrp(best_routes)
    
    for _ in range(1, max_restarts):
        routes, route_distances = perturb(best_routes, [compute_route_distance(r) for r in best_routes])
        routes, route_distances, current_max = local_search(routes, route_distances)
        if current_max < best_max_val - 1e-12:
            best_max_val = current_max
            best_routes = routes
            report_best_vrp(best_routes)
    
    return best_routes