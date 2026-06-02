import numpy as np
import random
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    depot = 0
    
    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def max_distance(routes):
        return max(route_distance(r) for r in routes)
    
    def copy_routes(routes):
        return [r[:] for r in routes]
    
    def feasibility(routes):
        # check each route starts and ends at 0, no duplicate customers, all customers covered
        seen = set()
        for r in routes:
            if r[0] != 0 or r[-1] != 0:
                return False
            for c in r[1:-1]:
                if c in seen:
                    return False
                seen.add(c)
        expected = set(range(1, n))
        return seen == expected
    
    best_routes = None
    best_max = float('inf')
    
    restarts = max(5, n // 10) if n > 10 else 5
    
    for restart in range(restarts):
        # initialize routes with depot
        routes = [[0, 0] for _ in range(truck_count)]
        customers = list(range(1, n))
        random.shuffle(customers)
        
        # regret-2 insertion (bias to minimize max distance)
        for cust in customers:
            best_max_after = float('inf')
            best_route = -1
            best_pos = -1
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    # compute new route distance if inserted at pos
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_route_dist = route_distance(new_route)
                    # compute new max distance
                    old_max = max_distance(routes)
                    other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx)
                    new_max = max(new_route_dist, other_max)
                    if new_max < best_max_after:
                        best_max_after = new_max
                        best_route = r_idx
                        best_pos = pos
            # insert at best position
            routes[best_route].insert(best_pos, cust)
        
        # initial best for this restart
        local_best_routes = copy_routes(routes)
        local_best_max = max_distance(routes)
        if local_best_max < best_max:
            best_max = local_best_max
            best_routes = copy_routes(routes)
            report_best_vrp(best_routes)
        
        # local search loop
        max_iter = (n - 1) * truck_count * 2
        no_improve = 0
        for iteration in range(max_iter):
            improved = False
            # order of operators: 2opt, relocate, swap, cross (randomized each iteration)
            operators = ['2opt', 'relocate', 'swap', 'cross']
            random.shuffle(operators)
            for op in operators:
                if op == '2opt':
                    # intra-route 2-opt
                    for r_idx in range(truck_count):
                        route = routes[r_idx]
                        if len(route) <= 3:
                            continue
                        for i in range(1, len(route)-2):
                            for j in range(i+1, len(route)-1):
                                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                                # compute new max
                                new_dist = route_distance(new_route)
                                other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != r_idx)
                                new_max = max(new_dist, other_max)
                                if new_max < local_best_max:
                                    routes[r_idx] = new_route
                                    local_best_max = new_max
                                    local_best_routes = copy_routes(routes)
                                    improved = True
                                    if local_best_max < best_max:
                                        best_max = local_best_max
                                        best_routes = copy_routes(routes)
                                        report_best_vrp(best_routes)
                                    # break to next iteration
                                    break
                            if improved:
                                break
                        if improved:
                            break
                elif op == 'relocate':
                    for src in range(truck_count):
                        if len(routes[src]) <= 2:
                            continue
                        for pos_src in range(1, len(routes[src])-1):
                            cust = routes[src][pos_src]
                            temp_src = routes[src][:pos_src] + routes[src][pos_src+1:]
                            dist_src = route_distance(temp_src)
                            for dst in range(truck_count):
                                if dst == src:
                                    continue
                                for pos_dst in range(1, len(routes[dst])):
                                    new_dst = routes[dst][:pos_dst] + [cust] + routes[dst][pos_dst:]
                                    dist_dst = route_distance(new_dst)
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != src and x != dst)
                                    new_max = max(dist_src, dist_dst, other_max)
                                    if new_max < local_best_max:
                                        routes[src] = temp_src
                                        routes[dst] = new_dst
                                        local_best_max = new_max
                                        local_best_routes = copy_routes(routes)
                                        improved = True
                                        if local_best_max < best_max:
                                            best_max = local_best_max
                                            best_routes = copy_routes(routes)
                                            report_best_vrp(best_routes)
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                elif op == 'swap':
                    for t1 in range(truck_count):
                        if len(routes[t1]) <= 2:
                            continue
                        for t2 in range(t1+1, truck_count):
                            if len(routes[t2]) <= 2:
                                continue
                            for i in range(1, len(routes[t1])-1):
                                for j in range(1, len(routes[t2])-1):
                                    cust1 = routes[t1][i]
                                    cust2 = routes[t2][j]
                                    new_route1 = routes[t1][:i] + [cust2] + routes[t1][i+1:]
                                    new_route2 = routes[t2][:j] + [cust1] + routes[t2][j+1:]
                                    dist1 = route_distance(new_route1)
                                    dist2 = route_distance(new_route2)
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
                                    new_max = max(dist1, dist2, other_max)
                                    if new_max < local_best_max:
                                        routes[t1] = new_route1
                                        routes[t2] = new_route2
                                        local_best_max = new_max
                                        local_best_routes = copy_routes(routes)
                                        improved = True
                                        if local_best_max < best_max:
                                            best_max = local_best_max
                                            best_routes = copy_routes(routes)
                                            report_best_vrp(best_routes)
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                elif op == 'cross':
                    for t1 in range(truck_count):
                        if len(routes[t1]) <= 2:
                            continue
                        for t2 in range(t1+1, truck_count):
                            if len(routes[t2]) <= 2:
                                continue
                            for i in range(1, len(routes[t1])-1):
                                for j in range(1, len(routes[t2])-1):
                                    new_route1 = routes[t1][:i] + routes[t2][j:]
                                    new_route2 = routes[t2][:j] + routes[t1][i:]
                                    dist1 = route_distance(new_route1)
                                    dist2 = route_distance(new_route2)
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
                                    new_max = max(dist1, dist2, other_max)
                                    if new_max < local_best_max:
                                        routes[t1] = new_route1
                                        routes[t2] = new_route2
                                        local_best_max = new_max
                                        local_best_routes = copy_routes(routes)
                                        improved = True
                                        if local_best_max < best_max:
                                            best_max = local_best_max
                                            best_routes = copy_routes(routes)
                                            report_best_vrp(best_routes)
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                if improved:
                    break
            if improved:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 3:
                    # perturbation: move 2 random customers to random routes
                    # select 2 distinct customers from routes (excluding depot)
                    all_custs = []
                    for r in routes:
                        all_custs.extend(r[1:-1])
                    if len(all_custs) >= 2:
                        to_move = random.sample(all_custs, 2)
                        for cust in to_move:
                            # find current route and remove
                            for r in routes:
                                if cust in r:
                                    r.remove(cust)
                                    break
                        # reinsert greedily as in construction
                        random.shuffle(to_move)
                        for cust in to_move:
                            best_max_after = float('inf')
                            best_route = -1
                            best_pos = -1
                            for r_idx in range(truck_count):
                                route = routes[r_idx]
                                for pos in range(1, len(route)):
                                    new_route = route[:pos] + [cust] + route[pos:]
                                    new_route_dist = route_distance(new_route)
                                    other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx)
                                    new_max = max(new_route_dist, other_max)
                                    if new_max < best_max_after:
                                        best_max_after = new_max
                                        best_route = r_idx
                                        best_pos = pos
                            routes[best_route].insert(best_pos, cust)
                        # reset local best
                        local_best_max = max_distance(routes)
                        local_best_routes = copy_routes(routes)
                        no_improve = 0
                    else:
                        break
        # end local search loop
        # update global best
        if local_best_max < best_max:
            best_max = local_best_max
            best_routes = copy_routes(local_best_routes)
            report_best_vrp(best_routes)
    
    # ensure best_routes is valid (should be)
    if best_routes is None:
        # fallback: empty routes
        best_routes = [[0, 0] for _ in range(truck_count)]
    return best_routes