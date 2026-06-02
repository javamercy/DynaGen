import numpy as np
import random
import math
import itertools
import collections

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    
    def route_distance(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def max_distance(routes):
        return max(route_distance(r) for r in routes)
    
    def decode(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        current_max = 0.0
        for cust in perm:
            best_increase = float('inf')
            best_route = -1
            best_pos = -1
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    added = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_route_dist = route_distance(route) + added
                    new_max = max(new_route_dist, max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx))
                    increase = new_max - current_max
                    if increase < best_increase or (math.isclose(increase, best_increase) and r_idx < best_route):
                        best_increase = increase
                        best_route = r_idx
                        best_pos = pos
            routes[best_route].insert(best_pos, cust)
            current_max = max_distance(routes)
        return routes
    
    def local_search(routes):
        best_routes = [r[:] for r in routes]
        best_max = max_distance(routes)
        report_best_vrp(best_routes)
        max_iter = (n - 1) * truck_count * 10
        for _ in range(max_iter):
            improved = False
            # Intra-route 2-opt
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
                        new_max = max(new_dist, max(route_distance(routes[x]) for x in range(truck_count) if x != r_idx))
                        if new_max < best_max:
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
            if improved:
                continue
            # Inter-route relocate
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
                            new_max = max(dist_src, dist_dst, max(route_distance(routes[x]) for x in range(truck_count) if x != src and x != dst))
                            if new_max < best_max:
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
            if improved:
                continue
            # Inter-route swap
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
                            new_max = max(dist1, dist2, max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2))
                            if new_max < best_max:
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
                continue
            # Cross-route 2-opt*
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
                            dist1 = route_distance(new_route1)
                            dist2 = route_distance(new_route2)
                            new_max = max(dist1, dist2, max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2))
                            if new_max < best_max:
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
            if not improved:
                break
        return best_routes, best_max
    
    pop_size = min(20, max(10, (n-1)//2))
    gen_count = min(50, max(10, (n-1)*2))
    # Initialize population
    population = []
    for _ in range(pop_size):
        perm = list(range(1, n))
        random.shuffle(perm)
        routes = decode(perm)
        maxd = max_distance(routes)
        population.append((perm, routes, maxd))
    population.sort(key=lambda x: x[2])
    best_ever = population[0]
    report_best_vrp(best_ever[1])
    
    for gen in range(gen_count):
        new_pop = []
        # Keep elite
        elite_size = max(1, pop_size // 5)
        new_pop.extend(population[:elite_size])
        # Generate offspring
        while len(new_pop) < pop_size:
            # Tournament selection
            idx1 = random.randint(0, pop_size-1)
            idx2 = random.randint(0, pop_size-1)
            p1 = population[idx1] if population[idx1][2] < population[idx2][2] else population[idx2]
            idx3 = random.randint(0, pop_size-1)
            idx4 = random.randint(0, pop_size-1)
            p2 = population[idx3] if population[idx3][2] < population[idx4][2] else population[idx4]
            # Order crossover
            perm1 = p1[0][:]
            perm2 = p2[0][:]
            cxpoint1 = random.randint(0, n-2)
            cxpoint2 = random.randint(cxpoint1+1, n-1)
            child_perm = [None] * (n-1)
            child_perm[cxpoint1:cxpoint2] = perm1[cxpoint1:cxpoint2]
            used = set(child_perm[cxpoint1:cxpoint2])
            fill_pos = 0
            for gene in perm2:
                if gene not in used:
                    while child_perm[fill_pos] is not None:
                        fill_pos += 1
                    child_perm[fill_pos] = gene
            # Mutation: swap two positions
            if random.random() < 0.2:
                i = random.randint(0, n-2)
                j = random.randint(0, n-2)
                child_perm[i], child_perm[j] = child_perm[j], child_perm[i]
            # Evaluate
            child_routes = decode(child_perm)
            child_max = max_distance(child_routes)
            new_pop.append((child_perm, child_routes, child_max))
        # Apply local search to best individual
        best_idx = min(range(len(new_pop)), key=lambda i: new_pop[i][2])
        best_perm, best_routes, best_max = new_pop[best_idx]
        # Restart loop inside local search
        max_restarts = max(1, (n-1)//10)
        for restart in range(max_restarts):
            new_routes, new_max = local_search([r[:] for r in best_routes])
            if new_max < best_max:
                best_routes = [r[:] for r in new_routes]
                best_max = new_max
                report_best_vrp(best_routes)
            else:
                # Random restart
                new_perm = list(range(1, n))
                random.shuffle(new_perm)
                new_routes = decode(new_perm)
                new_max = max_distance(new_routes)
                if new_max < best_max:
                    best_routes = new_routes
                    best_max = new_max
                    best_perm = new_perm
                    report_best_vrp(best_routes)
        new_pop[best_idx] = (best_perm, best_routes, best_max)
        # Update global best
        if best_max < best_ever[2]:
            best_ever = new_pop[best_idx]
            report_best_vrp(best_ever[1])
        # Sort for next generation
        new_pop.sort(key=lambda x: x[2])
        population = new_pop[:pop_size]
    return best_ever[1]