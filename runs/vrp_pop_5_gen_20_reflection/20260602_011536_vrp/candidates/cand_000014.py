import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)  # deterministic
    n = distance_matrix.shape[0]
    dist = distance_matrix.tolist()
    
    def route_distance(route):
        return sum(dist[route[i]][route[i+1]] for i in range(len(route)-1))
    
    def objective(routes):
        return max(route_distance(r) for r in routes)
    
    def copy_routes(routes):
        return [list(r) for r in routes]
    
    def initial_construct():
        routes = [[0,0] for _ in range(truck_count)]
        unassigned = list(range(1,n))
        while unassigned:
            best_max = float('inf')
            best_total = float('inf')
            best_node = None
            best_route = None
            best_pos = None
            for node in unassigned:
                for r in range(truck_count):
                    route = routes[r]
                    for pos in range(1, len(route)):
                        new_distance = 0
                        prev = route[0]
                        for k in range(1, len(route)):
                            if k == pos:
                                new_distance += dist[prev][node]
                                prev = node
                            new_distance += dist[prev][route[k]]
                            prev = route[k]
                        new_route_dist = new_distance
                        current_max = 0
                        for rr in range(truck_count):
                            if rr == r:
                                dd = new_route_dist
                            else:
                                dd = route_distance(routes[rr])
                            if dd > current_max:
                                current_max = dd
                        if current_max < best_max or (current_max == best_max and new_route_dist < best_total):
                            best_max = current_max
                            best_total = new_route_dist
                            best_node = node
                            best_route = r
                            best_pos = pos
            routes[best_route].insert(best_pos, best_node)
            unassigned.remove(best_node)
        return routes
    
    def local_search(routes):
        best_obj = objective(routes)
        max_passes = 20
        for _ in range(max_passes):
            improved = False
            # Relocate
            for node in range(1, n):
                curr_route = None
                curr_pos = None
                for r, route in enumerate(routes):
                    for pos, cust in enumerate(route):
                        if cust == node:
                            curr_route = r
                            curr_pos = pos
                            break
                    if curr_route is not None:
                        break
                if curr_route is None:
                    continue
                best_new_obj = float('inf')
                best_r = None
                best_pos = None
                for r in range(truck_count):
                    if r == curr_route:
                        continue
                    route = routes[r]
                    for pos in range(1, len(route)):
                        new_route_r = route[:pos] + [node] + route[pos:]
                        new_route_curr = routes[curr_route][:curr_pos] + routes[curr_route][curr_pos+1:]
                        if len(new_route_curr) < 2:
                            new_route_curr = [0,0]
                        new_routes = copy_routes(routes)
                        new_routes[r] = new_route_r
                        new_routes[curr_route] = new_route_curr
                        obj = objective(new_routes)
                        if obj < best_new_obj:
                            best_new_obj = obj
                            best_r = r
                            best_pos = pos
                if best_new_obj < best_obj:
                    routes[curr_route].pop(curr_pos)
                    if len(routes[curr_route]) < 2:
                        routes[curr_route] = [0,0]
                    routes[best_r].insert(best_pos, node)
                    best_obj = best_new_obj
                    improved = True
                    report_best_vrp(copy_routes(routes))
                    break
            if improved:
                continue
            # Swap
            for i in range(1, n):
                ri = None
                pi = None
                for r, route in enumerate(routes):
                    for p, cust in enumerate(route):
                        if cust == i:
                            ri = r
                            pi = p
                            break
                    if ri is not None:
                        break
                if ri is None:
                    continue
                for j in range(i+1, n):
                    rj = None
                    pj = None
                    for r, route in enumerate(routes):
                        for p, cust in enumerate(route):
                            if cust == j:
                                rj = r
                                pj = p
                                break
                        if rj is not None:
                            break
                    if rj is None or rj == ri:
                        continue
                    route_i_without = routes[ri][:pi] + routes[ri][pi+1:]
                    if len(route_i_without) < 2:
                        route_i_without = [0,0]
                    route_j_without = routes[rj][:pj] + routes[rj][pj+1:]
                    if len(route_j_without) < 2:
                        route_j_without = [0,0]
                    best_obj_swap = float('inf')
                    best_pos_i = None
                    best_pos_j = None
                    for pos_i in range(1, len(route_i_without)):
                        for pos_j in range(1, len(route_j_without)):
                            new_route_i = route_i_without[:pos_i] + [j] + route_i_without[pos_i:]
                            new_route_j = route_j_without[:pos_j] + [i] + route_j_without[pos_j:]
                            new_routes = copy_routes(routes)
                            new_routes[ri] = new_route_i
                            new_routes[rj] = new_route_j
                            obj = objective(new_routes)
                            if obj < best_obj_swap:
                                best_obj_swap = obj
                                best_pos_i = pos_i
                                best_pos_j = pos_j
                    if best_obj_swap < best_obj:
                        routes[ri] = route_i_without[:best_pos_i] + [j] + route_i_without[best_pos_i:]
                        routes[rj] = route_j_without[:best_pos_j] + [i] + route_j_without[best_pos_j:]
                        best_obj = best_obj_swap
                        improved = True
                        report_best_vrp(copy_routes(routes))
                        break
                if improved:
                    break
            if improved:
                continue
            # 2-opt within each route
            for r in range(truck_count):
                route = routes[r]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        other_max = 0
                        for rr in range(truck_count):
                            if rr != r:
                                d = route_distance(routes[rr])
                                if d > other_max:
                                    other_max = d
                        new_max = max(new_dist, other_max)
                        if new_max < best_obj:
                            routes[r] = new_route
                            best_obj = new_max
                            improved = True
                            report_best_vrp(copy_routes(routes))
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Cross-route 2-opt
            for r1 in range(truck_count):
                route1 = routes[r1]
                if len(route1) <= 3:
                    continue
                for r2 in range(r1+1, truck_count):
                    route2 = routes[r2]
                    if len(route2) <= 3:
                        continue
                    for a in range(1, len(route1)-2):
                        for b in range(1, len(route2)-2):
                            new_route1 = route1[:a+1] + route2[b+1:]
                            new_route2 = route2[:b+1] + route1[a+1:]
                            new_dist1 = route_distance(new_route1)
                            new_dist2 = route_distance(new_route2)
                            other_max = 0
                            for rr in range(truck_count):
                                if rr != r1 and rr != r2:
                                    d = route_distance(routes[rr])
                                    if d > other_max:
                                        other_max = d
                            new_max = max(new_dist1, new_dist2, other_max)
                            if new_max < best_obj:
                                routes[r1] = new_route1
                                routes[r2] = new_route2
                                best_obj = new_max
                                improved = True
                                report_best_vrp(copy_routes(routes))
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return routes
    
    def perturb(routes):
        # Remove a random customer from the route with maximum distance and insert into a random position in another route
        max_route_idx = max(range(truck_count), key=lambda r: route_distance(routes[r]))
        route = routes[max_route_idx]
        if len(route) <= 2:
            return routes
        # choose random customer from route (excluding depots)
        pos = random.randint(1, len(route)-2)
        node = route[pos]
        # remove
        new_route = route[:pos] + route[pos+1:]
        if len(new_route) < 2:
            new_route = [0,0]
        routes[max_route_idx] = new_route
        # insert into random route at random position
        r = random.randint(0, truck_count-1)
        # ensure not same route (if only one truck, insert back)
        if truck_count == 1:
            r = 0
        else:
            while r == max_route_idx:
                r = random.randint(0, truck_count-1)
        pos2 = random.randint(1, len(routes[r])-1)
        routes[r].insert(pos2, node)
        return routes
    
    # Initial construction
    best_routes = None
    best_obj = float('inf')
    
    routes = initial_construct()
    obj = objective(routes)
    if obj < best_obj:
        best_obj = obj
        best_routes = copy_routes(routes)
        report_best_vrp(best_routes)
    
    routes = local_search(routes)
    obj = objective(routes)
    if obj < best_obj:
        best_obj = obj
        best_routes = copy_routes(routes)
        report_best_vrp(best_routes)
    
    # Restart with perturbation
    for restart in range(2):
        perturbed = perturb(copy_routes(routes))
        routes = local_search(perturbed)
        obj = objective(routes)
        if obj < best_obj:
            best_obj = obj
            best_routes = copy_routes(routes)
            report_best_vrp(best_routes)
    
    return best_routes