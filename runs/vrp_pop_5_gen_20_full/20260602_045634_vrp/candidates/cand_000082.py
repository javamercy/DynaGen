import numpy as np
import random
import itertools

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    
    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def max_route_distance(routes):
        return max(route_distance(r) for r in routes)
    
    def repair(routes):
        for i, r in enumerate(routes):
            if r[0] != 0:
                routes[i] = [0] + r
            if r[-1] != 0:
                routes[i] = routes[i] + [0]
        return routes
    
    # Regret-based insertion: for each customer compute best and second best insertion cost increase across routes
    def construct():
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        # Sort customers by distance from depot descending for better spread
        cust_dist = [distance_matrix[0, c] for c in customers]
        customers = [c for _, c in sorted(zip(cust_dist, customers), reverse=True)]
        for cust in customers:
            best_inc = float('inf')
            second_best_inc = float('inf')
            best_route = None
            best_pos = None
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    inc = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    if inc < best_inc:
                        second_best_inc = best_inc
                        best_inc = inc
                        best_route = r_idx
                        best_pos = pos
                    elif inc < second_best_inc:
                        second_best_inc = inc
            # Insert at best position
            routes[best_route].insert(best_pos, cust)
        return routes
    
    # Local search: VND with first improvement, deterministic tie-breaking
    def local_search(routes):
        best_routes = [r[:] for r in routes]
        best_max = max_route_distance(routes)
        improved = True
        max_iter = (n - 1) * truck_count * 2
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            phases = ['2opt', 'relocate', 'swap', 'cross', '2optstar']
            random.shuffle(phases)
            for phase in phases:
                if phase == '2opt':
                    for r_idx in range(truck_count):
                        route = routes[r_idx]
                        if len(route) <= 3:
                            continue
                        for i in range(1, len(route)-2):
                            for j in range(i+1, len(route)-1):
                                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                                old_dist = route_distance(route)
                                new_dist = route_distance(new_route)
                                if new_dist >= old_dist:
                                    continue
                                other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != r_idx)
                                new_max = max(new_dist, other_max)
                                if new_max < best_max - 1e-12:
                                    routes[r_idx] = new_route
                                    best_routes = [r[:] for r in routes]
                                    best_max = new_max
                                    improved = True
                                    report_best_vrp(best_routes)
                                    break
                            if improved:
                                break
                        if improved:
                            break
                elif phase == 'relocate':
                    for src in range(truck_count):
                        route_src = routes[src]
                        if len(route_src) <= 2:
                            continue
                        for pos_src in range(1, len(route_src)-1):
                            cust = route_src[pos_src]
                            temp_src = route_src[:pos_src] + route_src[pos_src+1:]
                            dist_src = route_distance(temp_src)
                            for dst in range(truck_count):
                                if dst == src:
                                    continue
                                route_dst = routes[dst]
                                for pos_dst in range(1, len(route_dst)):
                                    new_dst = route_dst[:pos_dst] + [cust] + route_dst[pos_dst:]
                                    dist_dst = route_distance(new_dst)
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != src and x != dst)
                                    new_max = max(dist_src, dist_dst, other_max)
                                    if new_max < best_max - 1e-12:
                                        routes[src] = temp_src
                                        routes[dst] = new_dst
                                        best_routes = [r[:] for r in routes]
                                        best_max = new_max
                                        improved = True
                                        report_best_vrp(best_routes)
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                elif phase == 'swap':
                    for t1 in range(truck_count):
                        route1 = routes[t1]
                        if len(route1) <= 2:
                            continue
                        for t2 in range(t1+1, truck_count):
                            route2 = routes[t2]
                            if len(route2) <= 2:
                                continue
                            for i in range(1, len(route1)-1):
                                for j in range(1, len(route2)-1):
                                    cust1 = route1[i]
                                    cust2 = route2[j]
                                    new_route1 = route1[:i] + [cust2] + route1[i+1:]
                                    new_route2 = route2[:j] + [cust1] + route2[j+1:]
                                    dist1 = route_distance(new_route1)
                                    dist2 = route_distance(new_route2)
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
                                    new_max = max(dist1, dist2, other_max)
                                    if new_max < best_max - 1e-12:
                                        routes[t1] = new_route1
                                        routes[t2] = new_route2
                                        best_routes = [r[:] for r in routes]
                                        best_max = new_max
                                        improved = True
                                        report_best_vrp(best_routes)
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                elif phase == 'cross':
                    for t1 in range(truck_count):
                        route1 = routes[t1]
                        if len(route1) <= 2:
                            continue
                        for t2 in range(t1+1, truck_count):
                            route2 = routes[t2]
                            if len(route2) <= 2:
                                continue
                            for i in range(1, len(route1)-1):
                                for j in range(1, len(route2)-1):
                                    new_route1 = route1[:i] + route2[j:]
                                    new_route2 = route2[:j] + route1[i:]
                                    # Ensure depot at ends
                                    if new_route1[0] != 0:
                                        new_route1 = [0] + new_route1
                                    if new_route1[-1] != 0:
                                        new_route1.append(0)
                                    if new_route2[0] != 0:
                                        new_route2 = [0] + new_route2
                                    if new_route2[-1] != 0:
                                        new_route2.append(0)
                                    dist1 = route_distance(new_route1)
                                    dist2 = route_distance(new_route2)
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
                                    new_max = max(dist1, dist2, other_max)
                                    if new_max < best_max - 1e-12:
                                        routes[t1] = new_route1
                                        routes[t2] = new_route2
                                        best_routes = [r[:] for r in routes]
                                        best_max = new_max
                                        improved = True
                                        report_best_vrp(best_routes)
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                elif phase == '2optstar':
                    for t1 in range(truck_count):
                        route1 = routes[t1]
                        if len(route1) <= 2:
                            continue
                        for t2 in range(t1+1, truck_count):
                            route2 = routes[t2]
                            if len(route2) <= 2:
                                continue
                            for i in range(1, len(route1)-1):
                                for j in range(1, len(route2)-1):
                                    new_route1 = route1[:i] + route2[j:]
                                    new_route2 = route2[:j] + route1[i:]
                                    if new_route1[0] != 0:
                                        new_route1 = [0] + new_route1
                                    if new_route1[-1] != 0:
                                        new_route1.append(0)
                                    if new_route2[0] != 0:
                                        new_route2 = [0] + new_route2
                                    if new_route2[-1] != 0:
                                        new_route2.append(0)
                                    dist1 = route_distance(new_route1)
                                    dist2 = route_distance(new_route2)
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
                                    new_max = max(dist1, dist2, other_max)
                                    if new_max < best_max - 1e-12:
                                        routes[t1] = new_route1
                                        routes[t2] = new_route2
                                        best_routes = [r[:] for r in routes]
                                        best_max = new_max
                                        improved = True
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
        return best_routes
    
    # Adaptive perturbation: remove up to intensity customers from longest and second-longest routes, reinsert using regret
    def perturb(routes, intensity):
        if intensity == 0:
            return routes
        # Identify longest routes
        dists = [route_distance(r) for r in routes]
        sorted_indices = sorted(range(truck_count), key=lambda i: dists[i], reverse=True)
        # Remove from longest route
        longest_idx = sorted_indices[0]
        route = routes[longest_idx]
        if len(route) > 3:
            remove_count = min(intensity, len(route)-2)
            # Remove random block of length remove_count from the route (excluding depots)
            start = random.randint(1, len(route)-1-remove_count)
            removed_custs = route[start:start+remove_count]
            del route[start:start+remove_count]
        else:
            removed_custs = []
        # If intensity > 1 and there is a second longest route, remove from it as well
        if intensity > 1 and truck_count > 1:
            second_idx = sorted_indices[1]
            route2 = routes[second_idx]
            if len(route2) > 3:
                remove_count2 = min(intensity-1, len(route2)-2)
                start2 = random.randint(1, len(route2)-1-remove_count2)
                removed_custs2 = route2[start2:start2+remove_count2]
                del route2[start2:start2+remove_count2]
                removed_custs.extend(removed_custs2)
        # Reinsert removed customers using regret heuristic (as in construction but only for removed ones)
        random.shuffle(removed_custs)
        for cust in removed_custs:
            best_inc = float('inf')
            best_route = None
            best_pos = None
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    inc = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    if inc < best_inc:
                        best_inc = inc
                        best_route = r_idx
                        best_pos = pos
            routes[best_route].insert(best_pos, cust)
        return routes
    
    global_best_routes = None
    global_best_max = float('inf')
    restarts = max(5, n // 10)
    no_improve_restarts = 0
    for restart in range(restarts):
        routes = construct()
        routes = local_search(routes)
        current_max = max_route_distance(routes)
        if current_max < global_best_max - 1e-12:
            global_best_max = current_max
            global_best_routes = [r[:] for r in routes]
            no_improve_restarts = 0
        else:
            no_improve_restarts += 1
        intensity = 1 + (no_improve_restarts // 3)
        if restart < restarts - 1:
            # Use global best or current best as base
            if random.random() < 0.5 and global_best_routes is not None:
                base = [r[:] for r in global_best_routes]
            else:
                base = [r[:] for r in routes]
            routes = perturb(base, intensity)
    
    if global_best_routes is None:
        global_best_routes = [[0, 0] for _ in range(truck_count)]
    routes = repair(global_best_routes)
    return routes