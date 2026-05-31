import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    random.seed(0)

    # Helper: route distance
    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    # Regret construction (from parents)
    def regret_construction():
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        unassigned = list(range(1, n))
        def best_insertion(c):
            best = (float('inf'), -1, -1)
            second = (float('inf'), -1, -1)
            for r_idx, route in enumerate(routes):
                if len(route) < 2:
                    continue
                other_max = 0.0
                for j, d in enumerate(route_dists):
                    if j != r_idx and d > other_max:
                        other_max = d
                for pos in range(1, len(route)):
                    pred = route[pos-1]
                    succ = route[pos]
                    new_dist = route_dists[r_idx] - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
                    new_max = max(other_max, new_dist)
                    if new_max < best[0]:
                        best, second = (new_max, r_idx, pos), best
                    elif new_max < second[0]:
                        second = (new_max, r_idx, pos)
            return best[0], best[1], best[2], second[0]
        while unassigned:
            bests = []
            for c in unassigned:
                best_new_max, best_route, best_pos, second_new_max = best_insertion(c)
                if best_route == -1:
                    continue
                regret = second_new_max - best_new_max if second_new_max != float('inf') else float('inf')
                bests.append((-regret, c, best_route, best_pos, best_new_max))
            bests.sort(key=lambda x: (x[0], x[1]))
            _, c, best_route, best_pos, new_max = bests[0]
            route = routes[best_route]
            route.insert(best_pos, c)
            route_dists[best_route] = route_dist(route)
            unassigned.remove(c)
        return routes

    # Local search (from parents, simplified but effective)
    def improve(routes, route_dists):
        # Intra-route 2-opt
        for r_idx in range(truck_count):
            improved = True
            while improved:
                improved = False
                route = routes[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                        new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                        if new < old - 1e-12:
                            route[i:j+1] = reversed(route[i:j+1])
                            improved = True
                            route_dists[r_idx] = route_dist(route)
                            break
                    if improved:
                        break
        # Inter-route best-improvement relocate, swap, 2-opt* (bounded iterations)
        max_iter = n * truck_count
        for _ in range(max_iter):
            improved_overall = False
            # Relocate from longest route
            max_dist = max(route_dists)
            max_idx = route_dists.index(max_dist)
            best_move = None
            best_new_max = max_dist
            route_max = routes[max_idx]
            for i in range(1, len(route_max)-1):
                c = route_max[i]
                pred = route_max[i-1]
                succ = route_max[i+1]
                new_max_dist = route_dists[max_idx] - distance_matrix[pred, c] - distance_matrix[c, succ] + distance_matrix[pred, succ]
                for other_idx in range(truck_count):
                    if other_idx == max_idx:
                        continue
                    other_route = routes[other_idx]
                    for pos in range(1, len(other_route)):
                        pred_o = other_route[pos-1]
                        succ_o = other_route[pos]
                        new_other = route_dists[other_idx] - distance_matrix[pred_o, succ_o] + distance_matrix[pred_o, c] + distance_matrix[c, succ_o]
                        other_max = 0.0
                        for j, d in enumerate(route_dists):
                            if j != max_idx and j != other_idx and d > other_max:
                                other_max = d
                        new_overall = max(other_max, new_max_dist, new_other)
                        if new_overall < best_new_max - 1e-12:
                            best_new_max = new_overall
                            best_move = (i, other_idx, pos, new_max_dist, new_other)
            if best_move is not None:
                i, other_idx, pos, new_max_dist, new_other = best_move
                c = route_max.pop(i)
                routes[other_idx].insert(pos, c)
                route_dists[max_idx] = new_max_dist
                route_dists[other_idx] = new_other
                # Intra-2-opt on changed routes
                for r_idx in [max_idx, other_idx]:
                    improved = True
                    while improved:
                        improved = False
                        route = routes[r_idx]
                        for a in range(1, len(route)-2):
                            for b in range(a+1, len(route)-1):
                                old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                                new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                if new < old - 1e-12:
                                    route[a:b+1] = reversed(route[a:b+1])
                                    improved = True
                                    route_dists[r_idx] = route_dist(route)
                                    break
                            if improved:
                                break
                improved_overall = True
            # Swap
            if not improved_overall:
                max_dist = max(route_dists)
                max_idx = route_dists.index(max_dist)
                best_swap = None
                best_new_max = max_dist
                route_max = routes[max_idx]
                for i in range(1, len(route_max)-1):
                    c1 = route_max[i]
                    for other_idx in range(truck_count):
                        if other_idx == max_idx:
                            continue
                        other_route = routes[other_idx]
                        for j in range(1, len(other_route)-1):
                            c2 = other_route[j]
                            old1 = route_dists[max_idx]
                            old2 = route_dists[other_idx]
                            pred1 = route_max[i-1]
                            succ1 = route_max[i+1]
                            new_dist_max = old1 - distance_matrix[pred1, c1] - distance_matrix[c1, succ1] + distance_matrix[pred1, c2] + distance_matrix[c2, succ1]
                            pred2 = other_route[j-1]
                            succ2 = other_route[j+1]
                            new_dist_other = old2 - distance_matrix[pred2, c2] - distance_matrix[c2, succ2] + distance_matrix[pred2, c1] + distance_matrix[c1, succ2]
                            other_max = 0.0
                            for k, d in enumerate(route_dists):
                                if k != max_idx and k != other_idx and d > other_max:
                                    other_max = d
                            new_overall = max(other_max, new_dist_max, new_dist_other)
                            if new_overall < best_new_max - 1e-12:
                                best_new_max = new_overall
                                best_swap = (i, other_idx, j, new_dist_max, new_dist_other)
                if best_swap is not None:
                    i, other_idx, j, new_dist_max, new_dist_other = best_swap
                    route_max = routes[max_idx]
                    other_route = routes[other_idx]
                    c1 = route_max[i]
                    c2 = other_route[j]
                    route_max[i] = c2
                    other_route[j] = c1
                    route_dists[max_idx] = new_dist_max
                    route_dists[other_idx] = new_dist_other
                    for r_idx in [max_idx, other_idx]:
                        improved = True
                        while improved:
                            improved = False
                            route = routes[r_idx]
                            for a in range(1, len(route)-2):
                                for b in range(a+1, len(route)-1):
                                    old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                                    new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                    if new < old - 1e-12:
                                        route[a:b+1] = reversed(route[a:b+1])
                                        improved = True
                                        route_dists[r_idx] = route_dist(route)
                                        break
                                if improved:
                                    break
                    improved_overall = True
            # 2-opt*
            if not improved_overall:
                max_dist = max(route_dists)
                max_idx = route_dists.index(max_dist)
                best_cross = None
                best_new_max = max_dist
                route_max = routes[max_idx]
                for other_idx in range(truck_count):
                    if other_idx == max_idx:
                        continue
                    other_route = routes[other_idx]
                    for i in range(1, len(route_max)-1):
                        for j in range(1, len(other_route)-1):
                            if route_max[-1] != 0 or other_route[-1] != 0:
                                continue
                            old1 = distance_matrix[route_max[i], route_max[i+1]]
                            old2 = distance_matrix[other_route[j], other_route[j+1]]
                            new1 = distance_matrix[route_max[i], other_route[j+1]]
                            new2 = distance_matrix[other_route[j], route_max[i+1]]
                            new_dist_max = route_dists[max_idx] - old1 + new1
                            new_dist_other = route_dists[other_idx] - old2 + new2
                            other_max = 0.0
                            for k, d in enumerate(route_dists):
                                if k != max_idx and k != other_idx and d > other_max:
                                    other_max = d
                            new_overall = max(other_max, new_dist_max, new_dist_other)
                            if new_overall < best_new_max - 1e-12:
                                best_new_max = new_overall
                                best_cross = (i, other_idx, j, new_dist_max, new_dist_other)
                if best_cross is not None:
                    i, other_idx, j, new_dist_max, new_dist_other = best_cross
                    route_max = routes[max_idx]
                    other_route = routes[other_idx]
                    new_route_max = route_max[:i+1] + other_route[j+1:]
                    new_route_other = other_route[:j+1] + route_max[i+1:]
                    routes[max_idx] = new_route_max
                    routes[other_idx] = new_route_other
                    route_dists[max_idx] = route_dist(new_route_max)
                    route_dists[other_idx] = route_dist(new_route_other)
                    for r_idx in [max_idx, other_idx]:
                        improved = True
                        while improved:
                            improved = False
                            route = routes[r_idx]
                            for a in range(1, len(route)-2):
                                for b in range(a+1, len(route)-1):
                                    old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                                    new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                    if new < old - 1e-12:
                                        route[a:b+1] = reversed(route[a:b+1])
                                        improved = True
                                        route_dists[r_idx] = route_dist(route)
                                        break
                                if improved:
                                    break
                    improved_overall = True
            if not improved_overall:
                break
        return routes, route_dists

    # Decode permutation into routes (split using binary search on max distance)
    def decode_perm(perm):
        # Binary search to find minimal feasible max distance
        low = 0.0
        high = sum(distance_matrix[perm[i], perm[i+1]] for i in range(len(perm)-1)) + distance_matrix[0, perm[0]] + distance_matrix[perm[-1], 0]
        best_routes = None
        for _ in range(20):  # binary search iterations
            mid = (low + high) / 2
            ok, routes = try_split(perm, mid)
            if ok:
                best_routes = routes
                high = mid
            else:
                low = mid
        return best_routes

    def try_split(perm, max_dist):
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        # Greedy assignment: for each customer in order, try to add to current route if possible; else start new
        # We'll maintain current route index
        # Since we need exactly truck_count routes, we must plan: we can try to fill routes sequentially
        cur_route = 0
        for c in perm:
            placed = False
            for attempt in range(truck_count):
                r_idx = (cur_route + attempt) % truck_count
                route = routes[r_idx]
                # potential new distance if inserted at end before depot
                if len(route) == 2:
                    new_dist = distance_matrix[0, c] + distance_matrix[c, 0]
                else:
                    new_dist = route_dists[r_idx] - distance_matrix[route[-2], 0] + distance_matrix[route[-2], c] + distance_matrix[c, 0]
                if new_dist <= max_dist:
                    # insert before depot (position len(route)-1)
                    route.insert(len(route)-1, c)
                    route_dists[r_idx] = new_dist
                    cur_route = (r_idx + 1) % truck_count
                    placed = True
                    break
            if not placed:
                return False, None
        # After all customers, all routes must end at depot (already)
        return True, routes

    # GA parameters
    pop_size = 20
    generations = 10
    mutation_rate = 0.2
    elite_count = 2

    # Initialize population
    population = []
    # Add regret solutions
    for _ in range(pop_size // 2):
        routes = regret_construction()
        population.append(routes)
    # Add random permutations
    for _ in range(pop_size - len(population)):
        perm = list(range(1, n))
        random.shuffle(perm)
        routes = decode_perm(perm)
        if routes is not None:
            population.append(routes)
        else:
            # fallback to regret
            population.append(regret_construction())

    # Evaluate fitness (max distance)
    def evaluate(routes):
        return max(route_dist(r) for r in routes)

    # Local search all initial solutions
    for i in range(len(population)):
        routes = population[i]
        dists = [route_dist(r) for r in routes]
        routes, dists = improve(routes, dists)
        population[i] = routes
        report_best_vrp(routes)

    best_idx = min(range(len(population)), key=lambda i: evaluate(population[i]))
    best_routes = [route[:] for route in population[best_idx]]
    best_max = evaluate(best_routes)

    # GA loop
    for gen in range(generations):
        # Selection: tournament
        new_population = []
        # Elitism
        sorted_indices = sorted(range(len(population)), key=lambda i: evaluate(population[i]))
        for idx in sorted_indices[:elite_count]:
            new_population.append([route[:] for route in population[idx]])
        while len(new_population) < pop_size:
            # tournament
            i1 = random.randint(0, len(population)-1)
            i2 = random.randint(0, len(population)-1)
            parent1 = population[i1] if evaluate(population[i1]) < evaluate(population[i2]) else population[i2]
            i1 = random.randint(0, len(population)-1)
            i2 = random.randint(0, len(population)-1)
            parent2 = population[i1] if evaluate(population[i1]) < evaluate(population[i2]) else population[i2]
            # Crossover: OX on permutation
            # Convert routes to permutation (order of first appearance, then second, etc.)
            def routes_to_perm(routes):
                perm = []
                seen = set()
                for r in routes:
                    for c in r:
                        if c != 0 and c not in seen:
                            perm.append(c)
                            seen.add(c)
                return perm
            perm1 = routes_to_perm(parent1)
            perm2 = routes_to_perm(parent2)
            # OX
            size = len(perm1)
            a = random.randint(0, size-1)
            b = random.randint(a, size-1)
            child_perm = [None] * size
            # copy segment from parent1
            child_perm[a:b+1] = perm1[a:b+1]
            # fill from parent2 in order
            pos = (b+1) % size
            for c in perm2:
                if c not in child_perm:
                    child_perm[pos] = c
                    pos = (pos+1) % size
            # Mutation: swap two random customers
            if random.random() < mutation_rate:
                i = random.randint(0, size-1)
                j = random.randint(0, size-1)
                child_perm[i], child_perm[j] = child_perm[j], child_perm[i]
            # Decode
            child_routes = decode_perm(child_perm)
            if child_routes is None:
                child_routes = regret_construction()
            # Local search
            child_dists = [route_dist(r) for r in child_routes]
            child_routes, child_dists = improve(child_routes, child_dists)
            new_population.append(child_routes)
            report_best_vrp(child_routes)
        # Replace population
        population = new_population
        # Update best
        for routes in population:
            cur_max = evaluate(routes)
            if cur_max < best_max - 1e-12:
                best_max = cur_max
                best_routes = [route[:] for route in routes]
                report_best_vrp(best_routes)

    # Ensure each route starts and ends at 0, no missing customers
    seen = set()
    for r in best_routes:
        assert r[0] == 0 and r[-1] == 0
        for c in r[1:-1]:
            seen.add(c)
    assert len(seen) == n-1
    if len(best_routes) != truck_count:
        # pad if missing (shouldn't happen)
        while len(best_routes) < truck_count:
            best_routes.append([0,0])
    return best_routes