import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    random.seed(0)
    
    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def total_dist(routes):
        return sum(route_dist(r) for r in routes)
    
    # DP to split a permutation into routes minimizing max route distance
    def split_permutation(perm):
        m = len(perm)
        if m == 0:
            return [[0,0] for _ in range(truck_count)]
        seg_dist = [[0]*m for _ in range(m)]
        for i in range(m):
            for j in range(i, m):
                if i == j:
                    seg_dist[i][j] = distance_matrix[0, perm[i]] + distance_matrix[perm[i], 0]
                else:
                    seg_dist[i][j] = distance_matrix[0, perm[i]]  # fix: already covered in i part? Actually we need total segment distance
        # Compute segment distances properly: distance from depot to first, along customers, back to depot
        for i in range(m):
            for j in range(i, m):
                d = distance_matrix[0, perm[i]]
                for k in range(i, j):
                    d += distance_matrix[perm[k], perm[k+1]]
                d += distance_matrix[perm[j], 0]
                seg_dist[i][j] = d
        # DP for minmax
        INF = 1e100
        dp = [[INF]*(truck_count+1) for _ in range(m+1)]
        dp[0][0] = 0.0
        choice = [[-1]*(truck_count+1) for _ in range(m+1)]
        for k in range(1, truck_count+1):
            for i in range(1, m+1):
                best = INF
                best_j = -1
                for j in range(k-1, i):
                    if dp[j][k-1] < INF:
                        cand = max(dp[j][k-1], seg_dist[j][i-1])  # seg_dist indices: from j to i-1 inclusive
                        if cand < best:
                            best = cand
                            best_j = j
                dp[i][k] = best
                choice[i][k] = best_j
        # Reconstruct routes
        routes = []
        i = m
        k = truck_count
        while k > 0:
            j = choice[i][k]
            if j == -1:
                break
            seg = perm[j:i]
            route = [0] + seg + [0]
            routes.append(route)
            i = j
            k -= 1
        # If we didn't use all trucks, fill with empty routes
        while len(routes) < truck_count:
            routes.append([0,0])
        return routes
    
    # Improvement function (same as parent but with nonlocal best references)
    def improve(routes, route_dists, best_routes, best_max, best_total):
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
        max_iter_local = n * truck_count
        for _ in range(max_iter_local):
            improved_overall = False
            # Best-improvement relocate with tie-breaking on total
            max_dist = max(route_dists)
            max_idx = route_dists.index(max_dist)
            best_move = None
            best_new_max = max_dist
            best_new_total = total_dist(routes)
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
                            best_new_total = best_total - route_dists[max_idx] - route_dists[other_idx] + new_max_dist + new_other
                            best_move = (i, other_idx, pos, new_max_dist, new_other)
                        elif abs(new_overall - best_new_max) < 1e-12:
                            new_total = best_total - route_dists[max_idx] - route_dists[other_idx] + new_max_dist + new_other
                            if new_total < best_new_total - 1e-12:
                                best_new_max = new_overall
                                best_new_total = new_total
                                best_move = (i, other_idx, pos, new_max_dist, new_other)
            if best_move is not None:
                i, other_idx, pos, new_max_dist, new_other = best_move
                c = route_max.pop(i)
                routes[other_idx].insert(pos, c)
                route_dists[max_idx] = new_max_dist
                route_dists[other_idx] = new_other
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
                cur_max = max(route_dists)
                cur_total = total_dist(routes)
                if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
                    best_max = cur_max
                    best_total = cur_total
                    best_routes[:] = [route[:] for route in routes]
                    report_best_vrp(best_routes)
                improved_overall = True
            if not improved_overall:
                # Best swap with tie-breaking on total
                max_dist = max(route_dists)
                max_idx = route_dists.index(max_dist)
                best_swap = None
                best_new_max = max_dist
                best_new_total = total_dist(routes)
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
                                best_new_total = best_total - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                                best_swap = (i, other_idx, j, new_dist_max, new_dist_other)
                            elif abs(new_overall - best_new_max) < 1e-12:
                                new_total = best_total - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                                if new_total < best_new_total - 1e-12:
                                    best_new_max = new_overall
                                    best_new_total = new_total
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
                    cur_max = max(route_dists)
                    cur_total = total_dist(routes)
                    if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
                        best_max = cur_max
                        best_total = cur_total
                        best_routes[:] = [route[:] for route in routes]
                        report_best_vrp(best_routes)
                    improved_overall = True
            if not improved_overall:
                # Best 2-opt* with tie-breaking on total
                max_dist = max(route_dists)
                max_idx = route_dists.index(max_dist)
                best_cross = None
                best_new_max = max_dist
                best_new_total = total_dist(routes)
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
                                best_new_total = best_total - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                                best_cross = (i, other_idx, j, new_dist_max, new_dist_other)
                            elif abs(new_overall - best_new_max) < 1e-12:
                                new_total = best_total - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                                if new_total < best_new_total - 1e-12:
                                    best_new_max = new_overall
                                    best_new_total = new_total
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
                    cur_max = max(route_dists)
                    cur_total = total_dist(routes)
                    if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
                        best_max = cur_max
                        best_total = cur_total
                        best_routes[:] = [route[:] for route in routes]
                        report_best_vrp(best_routes)
                    improved_overall = True
            if not improved_overall:
                break
        return routes, route_dists, best_routes, best_max, best_total
    
    # Helper to convert routes to permutation (concatenating routes, ignoring depot)
    def routes_to_perm(routes):
        perm = []
        for r in routes:
            if len(r) > 2:
                perm.extend(r[1:-1])
        return perm
    
    # Crossover: Order Crossover (OX)
    def crossover(p1, p2):
        m = len(p1)
        start = random.randint(0, m-2)
        end = random.randint(start+1, m-1)
        child = [None]*m
        child[start:end+1] = p1[start:end+1]
        remaining = [c for c in p2 if c not in child]
        idx = 0
        for i in range(m):
            if child[i] is None:
                child[i] = remaining[idx]
                idx += 1
        return child
    
    # Mutation: swap two random positions
    def mutate(perm):
        if len(perm) < 2:
            return perm
        i = random.randint(0, len(perm)-1)
        j = random.randint(0, len(perm)-1)
        perm[i], perm[j] = perm[j], perm[i]
        return perm
    
    # Initial population: best from greedy + random
    pop_size = 10
    generations = 15
    population = []
    best_overall_routes = None
    best_overall_max = float('inf')
    best_overall_total = float('inf')
    
    # Create initial solutions from regret construction (similar to parent)
    def construct_solution():
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        unassigned = list(range(1, n))
        while unassigned:
            bests = []
            for c in unassigned:
                # Simplified best insertion (without second best for regret, just use best max)
                best = (float('inf'), -1, -1)
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
                            best = (new_max, r_idx, pos)
                if best[1] == -1:
                    continue
                bests.append((best[0], c, best[1], best[2]))
            bests.sort(key=lambda x: (x[0], x[1]))
            _, c, r, pos = bests[0]
            routes[r].insert(pos, c)
            route_dists[r] = route_dist(routes[r])
            unassigned.remove(c)
        return routes, route_dists
    
    # Generate initial population: one from greedy and rest random permutations
    greedy_routes, _ = construct_solution()
    greedy_perm = routes_to_perm(greedy_routes)
    # Ensure permutation has all customers
    if len(greedy_perm) != n-1:
        greedy_perm = list(range(1,n))
    population.append(greedy_perm[:])
    for _ in range(pop_size-1):
        perm = list(range(1,n))
        random.shuffle(perm)
        population.append(perm)
    
    # Evaluate initial population with DP and local search
    for idx, perm in enumerate(population):
        routes = split_permutation(perm)
        route_dists = [route_dist(r) for r in routes]
        cur_max = max(route_dists)
        cur_total = total_dist(routes)
        best_routes_local = [r[:] for r in routes]
        best_max_local = cur_max
        best_total_local = cur_total
        if best_max_local < best_overall_max - 1e-12 or (abs(best_max_local - best_overall_max) < 1e-12 and best_total_local < best_overall_total - 1e-12):
            best_overall_max = best_max_local
            best_overall_total = best_total_local
            best_overall_routes = [r[:] for r in best_routes_local]
            report_best_vrp(best_overall_routes)
        # Apply local search
        routes, route_dists, best_routes_local, best_max_local, best_total_local = improve(
            routes, route_dists, best_routes_local, best_max_local, best_total_local)
        cur_max = max(route_dists)
        cur_total = total_dist(routes)
        if cur_max < best_max_local - 1e-12 or (abs(cur_max - best_max_local) < 1e-12 and cur_total < best_total_local - 1e-12):
            best_max_local = cur_max
            best_total_local = cur_total
            best_routes_local = [r[:] for r in routes]
        if best_max_local < best_overall_max - 1e-12 or (abs(best_max_local - best_overall_max) < 1e-12 and best_total_local < best_overall_total - 1e-12):
            best_overall_max = best_max_local
            best_overall_total = best_total_local
            best_overall_routes = [r[:] for r in best_routes_local]
            report_best_vrp(best_overall_routes)
        # Update permutation with improved routes
        new_perm = routes_to_perm(routes)
        if len(new_perm) == n-1:
            population[idx] = new_perm
    
    # Genetic algorithm loop
    for gen in range(generations):
        new_population = []
        # Elitism: keep best solution
        new_population.append(population[0] if len(population) > 0 else list(range(1,n)))
        while len(new_population) < pop_size:
            # Tournament selection
            t1 = random.randint(0, pop_size-1)
            t2 = random.randint(0, pop_size-1)
            parent1 = population[t1] if random.uniform(0,1) < 0.75 else population[t2]  # simple tournament
            t1 = random.randint(0, pop_size-1)
            t2 = random.randint(0, pop_size-1)
            parent2 = population[t1] if random.uniform(0,1) < 0.75 else population[t2]
            child_perm = crossover(parent1, parent2)
            if random.uniform(0,1) < 0.3:
                child_perm = mutate(child_perm)
            # Decode and improve
            routes = split_permutation(child_perm)
            route_dists = [route_dist(r) for r in routes]
            best_routes_local = [r[:] for r in routes]
            best_max_local = max(route_dists)
            best_total_local = total_dist(routes)
            routes, route_dists, best_routes_local, best_max_local, best_total_local = improve(
                routes, route_dists, best_routes_local, best_max_local, best_total_local)
            cur_max = max(route_dists)
            cur_total = total_dist(routes)
            if cur_max < best_max_local - 1e-12 or (abs(cur_max - best_max_local) < 1e-12 and cur_total < best_total_local - 1e-12):
                best_max_local = cur_max
                best_total_local = cur_total
                best_routes_local = [r[:] for r in routes]
            if best_max_local < best_overall_max - 1e-12 or (abs(best_max_local - best_overall_max) < 1e-12 and best_total_local < best_overall_total - 1e-12):
                best_overall_max = best_max_local
                best_overall_total = best_total_local
                best_overall_routes = [r[:] for r in best_routes_local]
                report_best_vrp(best_overall_routes)
            # Update permutation for population
            new_perm = routes_to_perm(routes)
            if len(new_perm) == n-1:
                new_population.append(new_perm)
            else:
                # if something went wrong, append random
                new_population.append(list(range(1,n)))
                random.shuffle(new_population[-1])
        population = new_population
        # Optional: sort population by solution quality (but not needed as we only use elite and tournament)
    
    return best_overall_routes