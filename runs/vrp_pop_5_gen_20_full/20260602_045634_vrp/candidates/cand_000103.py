import numpy as np
import random
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]

    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_distance(routes):
        return max(route_distance(r) for r in routes)

    def repair_routes(routes):
        for i, r in enumerate(routes):
            if r[0] != 0:
                routes[i] = [0] + r
            if r[-1] != 0:
                routes[i] = routes[i] + [0]
        return routes

    def perturb(routes):
        # Move 1 to max(2, n//10) customers, with bias from longest to shortest
        long_idx = max(range(truck_count), key=lambda i: route_distance(routes[i]))
        short_idx = min(range(truck_count), key=lambda i: route_distance(routes[i]))
        moves = random.randint(1, max(2, n // 10))
        for _ in range(moves):
            # choose source: biased to longest route
            if random.random() < 0.4:
                src = long_idx
            else:
                src = random.randint(0, truck_count-1)
            while len(routes[src]) <= 2:
                src = random.randint(0, truck_count-1)
            pos = random.randint(1, len(routes[src])-2)
            cust = routes[src].pop(pos)
            # choose destination: biased to shortest route
            if random.random() < 0.4:
                dst = short_idx
            else:
                dst = random.randint(0, truck_count-1)
            pos2 = random.randint(1, len(routes[dst])-1)
            routes[dst].insert(pos2, cust)
        return routes

    def local_search(routes, best_max):
        improved = True
        max_iter = (n - 1) * truck_count * 5
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            phases = ['2opt', 'relocate', 'swap', 'cross']
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
                                if new_max < best_max:
                                    routes[r_idx] = new_route
                                    best_max = new_max
                                    improved = True
                                    report_best_vrp([r[:] for r in routes])
                                    break
                            if improved:
                                break
                        if improved:
                            break
                elif phase == 'relocate':
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
                                route_dst = routes[dst]
                                for pos_dst in range(1, len(route_dst)):
                                    new_dst = route_dst[:pos_dst] + [cust] + route_dst[pos_dst:]
                                    dist_dst = route_distance(new_dst)
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != src and x != dst)
                                    new_max = max(dist_src, dist_dst, other_max)
                                    if new_max < best_max:
                                        routes[src] = temp_src
                                        routes[dst] = new_dst
                                        best_max = new_max
                                        improved = True
                                        report_best_vrp([r[:] for r in routes])
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
                                    if new_max < best_max:
                                        routes[t1] = new_route1
                                        routes[t2] = new_route2
                                        best_max = new_max
                                        improved = True
                                        report_best_vrp([r[:] for r in routes])
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
                                    dist1 = route_distance(new_route1)
                                    dist2 = route_distance(new_route2)
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
                                    new_max = max(dist1, dist2, other_max)
                                    if new_max < best_max:
                                        routes[t1] = new_route1
                                        routes[t2] = new_route2
                                        best_max = new_max
                                        improved = True
                                        report_best_vrp([r[:] for r in routes])
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                if improved:
                    break
        return routes, best_max

    # Constructive: greedy insertion minimizing max distance increase with random tie-breaking
    customers = list(range(1, n))
    random.shuffle(customers)
    routes = [[0, 0] for _ in range(truck_count)]
    for cust in customers:
        best_increase = float('inf')
        candidates = []
        current_max = max_distance(routes)
        for r_idx in range(truck_count):
            route = routes[r_idx]
            for pos in range(1, len(route)):
                prev = route[pos-1]
                nxt = route[pos]
                added = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                new_route_dist = route_distance(route) + added
                other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx)
                new_max = max(new_route_dist, other_max)
                increase = new_max - current_max
                if increase < best_increase:
                    best_increase = increase
                    candidates = [(r_idx, pos)]
                elif increase == best_increase:
                    candidates.append((r_idx, pos))
        best_route, best_pos = random.choice(candidates)
        routes[best_route].insert(best_pos, cust)

    best_routes = [r[:] for r in routes]
    best_max = max_distance(routes)
    report_best_vrp(best_routes)

    # Iterative improvement with perturbation
    max_iters = max(10, n // 2)
    for iteration in range(max_iters):
        # Perturb best solution
        new_routes = perturb([r[:] for r in best_routes])
        # Local search on perturbed solution
        new_routes, new_max = local_search(new_routes, best_max)
        if new_max < best_max:
            best_routes = [r[:] for r in new_routes]
            best_max = new_max
            report_best_vrp(best_routes)

    # Ensure exactly truck_count routes, each starts and ends with 0
    best_routes = repair_routes(best_routes)
    # Remove duplicate customers (should not happen, but safe)
    seen = set()
    for r in best_routes:
        for i, node in enumerate(r):
            if node != 0:
                if node in seen:
                    # replace with a dummy removal; but generally this won't occur
                    r[i] = 0
                else:
                    seen.add(node)
    # Ensure all customers 1..n-1 are present
    all_customers = set(range(1, n))
    present = set()
    for r in best_routes:
        for node in r:
            if node != 0:
                present.add(node)
    missing = all_customers - present
    # In case missing (shouldn't happen), add them as new routes or append to existing
    if missing:
        for cust in missing:
            # find route with smallest max distance increase
            best_inc = float('inf')
            best_r = None
            best_pos = None
            for r_idx in range(truck_count):
                route = best_routes[r_idx]
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    added = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    if added < best_inc:
                        best_inc = added
                        best_r = r_idx
                        best_pos = pos
            best_routes[best_r].insert(best_pos, cust)

    return best_routes