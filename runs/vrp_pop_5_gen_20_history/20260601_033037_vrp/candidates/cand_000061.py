import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_distance(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    best_routes = None
    best_max = float('inf')

    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        new_max = max(route_distance(r) for r in routes)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [list(r) for r in routes]

    # Decode permutation to routes: split into truck_count segments
    def decode(perm):
        n_cust = len(perm)
        base = n_cust // truck_count
        rem = n_cust % truck_count
        routes = []
        start = 0
        for i in range(truck_count):
            size = base + (1 if i < rem else 0)
            if size == 0:
                routes.append([0, 0])
            else:
                segment = perm[start:start+size]
                routes.append([0] + segment + [0])
                start += size
        return routes

    # Nearest neighbor permutation starting from a given customer
    def nearest_neighbor_perm(start):
        visited = set()
        perm = []
        curr = start
        visited.add(curr)
        perm.append(curr)
        while len(visited) < len(customers):
            candidates = [c for c in customers if c not in visited]
            next_node = min(candidates, key=lambda x: distance_matrix[curr, x])
            visited.add(next_node)
            perm.append(next_node)
            curr = next_node
        return perm

    # Initialize population
    pop_size = min(20, len(customers))
    population = []
    for i in range(pop_size):
        start = customers[i % len(customers)]
        perm = nearest_neighbor_perm(start)
        routes = decode(perm)
        max_dist = max(route_distance(r) for r in routes)
        population.append((perm, routes, max_dist))
        report_best_vrp(routes)

    # Sort population by max distance (ascending)
    population.sort(key=lambda x: x[2])

    # Local search on a route set (limited iterations)
    def local_search(routes, max_iter=5):
        # Relocate and swap from longest route, then 2-opt, limited iterations
        for _ in range(max_iter):
            improved = False
            dists = [route_distance(r) for r in routes]
            max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
            interior = routes[max_idx][1:-1]
            if not interior:
                break
            # Relocate from longest route
            for cust in interior:
                for other_idx in range(truck_count):
                    if other_idx == max_idx:
                        continue
                    other_route = routes[other_idx]
                    best_pos = None
                    best_delta = float('inf')
                    for pos in range(1, len(other_route)):
                        prev = other_route[pos-1]
                        nxt = other_route[pos] if pos < len(other_route) else 0
                        delta = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        if delta < best_delta - 1e-12:
                            best_delta = delta
                            best_pos = pos
                    new_routes = [list(r) for r in routes]
                    new_routes[max_idx].remove(cust)
                    new_routes[other_idx].insert(best_pos, cust)
                    new_max = max(route_distance(r) for r in new_routes)
                    if new_max < best_max - 1e-12:
                        routes = new_routes
                        report_best_vrp(routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # Swap between longest and another route
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_interior = routes[other_idx][1:-1]
                if not other_interior:
                    continue
                for cust_max in interior:
                    for cust_other in other_interior:
                        new_routes = [list(r) for r in routes]
                        idx_max = new_routes[max_idx].index(cust_max)
                        idx_other = new_routes[other_idx].index(cust_other)
                        new_routes[max_idx][idx_max] = cust_other
                        new_routes[other_idx][idx_other] = cust_max
                        new_max = max(route_distance(r) for r in new_routes)
                        if new_max < best_max - 1e-12:
                            routes = new_routes
                            report_best_vrp(routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # 2-opt on each route
            for idx in range(truck_count):
                route = routes[idx]
                if len(route) <= 3:
                    continue
                best_route = route[:]
                best_dist = route_distance(route)
                found = False
                for a in range(1, len(route)-2):
                    for b in range(a+1, len(route)-1):
                        new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                        new_dist = route_distance(new_route)
                        if new_dist < best_dist - 1e-12:
                            best_dist = new_dist
                            best_route = new_route
                            found = True
                            break
                    if found:
                        break
                if found:
                    routes[idx] = best_route
                    new_max = max(route_distance(r) for r in routes)
                    if new_max < best_max - 1e-12:
                        report_best_vrp(routes)
                    improved = True
                    break
            if not improved:
                break
        return routes

    # Main GA loop
    generations = min(100, n * truck_count)
    for gen in range(generations):
        # Selection: pick best two parents
        parent1 = population[0][0]
        parent2 = population[1][0]
        # One-point crossover with point based on generation
        point = (gen % len(customers)) + 1
        child_perm = list(parent1[:point])
        for c in parent2:
            if c not in child_perm:
                child_perm.append(c)
        # Decode child
        child_routes = decode(child_perm)
        # Apply local search to child
        child_routes = local_search(child_routes, max_iter=5)
        child_max = max(route_distance(r) for r in child_routes)
        report_best_vrp(child_routes)
        # Replacement: replace worst if child is better
        if child_max < population[-1][2] - 1e-12:
            population[-1] = (child_perm, child_routes, child_max)
            population.sort(key=lambda x: x[2])
        # Early stop if best is 0? Not needed

    return best_routes if best_routes is not None else population[0][1]