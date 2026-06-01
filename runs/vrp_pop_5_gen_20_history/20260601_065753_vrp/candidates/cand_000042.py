import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    def compute_route_length(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def decode(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        lengths = [0.0] * truck_count
        for cust in perm:
            best_max = float('inf')
            best_r = -1
            best_p = -1
            for r in range(truck_count):
                route = routes[r]
                for p in range(1, len(route)):
                    prev = route[p-1]
                    nxt = route[p]
                    new_len = lengths[r] - distance_matrix[prev, nxt] + distance_matrix[prev, cust] + distance_matrix[cust, nxt]
                    new_max = new_len
                    for rr in range(truck_count):
                        if rr != r and lengths[rr] > new_max:
                            new_max = lengths[rr]
                    if new_max < best_max:
                        best_max = new_max
                        best_r = r
                        best_p = p
                    elif new_max == best_max:
                        if r < best_r or (r == best_r and p < best_p):
                            best_max = new_max
                            best_r = r
                            best_p = p
            routes[best_r].insert(best_p, cust)
            lengths[best_r] = compute_route_length(routes[best_r])
        max_len = max(lengths)
        return routes, max_len
    
    def report_best_vrp(routes):
        nonlocal best_max, best_routes
        m = max(compute_route_length(r) for r in routes)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]
    
    def greedy_construction():
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        unassigned = set(customers)
        while unassigned:
            candidates = []
            for cust in list(unassigned):
                best_inc = float('inf')
                best_t = -1
                best_p = -1
                for t in range(truck_count):
                    route = routes[t]
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_dist = compute_route_length(new_route)
                        new_max = new_dist
                        for k in range(truck_count):
                            if k != t:
                                new_max = max(new_max, route_dists[k])
                        if new_max < best_inc:
                            best_inc = new_max
                            best_t = t
                            best_p = pos
                        elif new_max == best_inc:
                            if cust < best_inc or (cust == best_inc and (t < best_t or (t == best_t and pos < best_p))):
                                best_inc = new_max
                                best_t = t
                                best_p = pos
                candidates.append((best_inc, cust, best_t, best_p))
            candidates.sort(key=lambda x: (x[0], x[1]))
            _, cust, t, pos = candidates[0]
            routes[t] = routes[t][:pos] + [cust] + routes[t][pos:]
            route_dists[t] = compute_route_length(routes[t])
            unassigned.remove(cust)
        # Convert to permutation
        perm = []
        for r in routes:
            perm.extend(r[1:-1])
        return perm, routes, max(route_dists)
    
    # Initialize population
    pop_size = min(50, n)
    max_gen = 5 * n
    mutation_prob_start = 0.25
    mutation_prob_end = 0.05
    
    population = []
    best_max = float('inf')
    best_routes = None
    
    # Greedy seed
    greedy_perm, greedy_routes, greedy_max = greedy_construction()
    population.append((greedy_max, greedy_perm))
    report_best_vrp(greedy_routes)
    
    for _ in range(pop_size - 1):
        perm = customers[:]
        random.shuffle(perm)
        routes, max_len = decode(perm)
        population.append((max_len, perm))
        report_best_vrp(routes)
    population.sort(key=lambda x: x[0])
    
    def local_search(perm):
        routes, max_len = decode(perm)
        improved = True
        max_iter = (n - 1) * truck_count * 2
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            best_delta = 0
            best_move = None
            # Relocate
            for t1 in range(truck_count):
                route1 = routes[t1]
                if len(route1) <= 2:
                    continue
                for i in range(1, len(route1)-1):
                    cust = route1[i]
                    new_route1 = route1[:i] + route1[i+1:]
                    dist1_new = compute_route_length(new_route1)
                    for t2 in range(truck_count):
                        if t1 == t2:
                            continue
                        route2 = routes[t2]
                        for j in range(1, len(route2)):
                            new_route2 = route2[:j] + [cust] + route2[j:]
                            dist2_new = compute_route_length(new_route2)
                            current_max = max_len
                            new_max = max(dist1_new, dist2_new)
                            for k in range(truck_count):
                                if k != t1 and k != t2:
                                    new_max = max(new_max, compute_route_length(routes[k]))
                            if new_max < current_max:
                                delta = current_max - new_max
                                if delta > best_delta:
                                    best_delta = delta
                                    best_move = ('relocate', t1, i, t2, j)
            # Swap
            for t1 in range(truck_count):
                route1 = routes[t1]
                if len(route1) <= 2:
                    continue
                for i in range(1, len(route1)-1):
                    cust1 = route1[i]
                    for t2 in range(t1+1, truck_count):
                        route2 = routes[t2]
                        if len(route2) <= 2:
                            continue
                        for j in range(1, len(route2)-1):
                            cust2 = route2[j]
                            new_route1 = route1[:i] + [cust2] + route1[i+1:]
                            new_route2 = route2[:j] + [cust1] + route2[j+1:]
                            dist1_new = compute_route_length(new_route1)
                            dist2_new = compute_route_length(new_route2)
                            current_max = max_len
                            new_max = max(dist1_new, dist2_new)
                            for k in range(truck_count):
                                if k != t1 and k != t2:
                                    new_max = max(new_max, compute_route_length(routes[k]))
                            if new_max < current_max:
                                delta = current_max - new_max
                                if delta > best_delta:
                                    best_delta = delta
                                    best_move = ('swap', t1, i, t2, j)
            if best_move:
                improved = True
                if best_move[0] == 'relocate':
                    _, t1, i, t2, j = best_move
                    cust = routes[t1][i]
                    routes[t1] = routes[t1][:i] + routes[t1][i+1:]
                    routes[t2] = routes[t2][:j] + [cust] + routes[t2][j:]
                else:
                    _, t1, i, t2, j = best_move
                    cust1 = routes[t1][i]
                    cust2 = routes[t2][j]
                    routes[t1][i] = cust2
                    routes[t2][j] = cust1
                max_len = max(compute_route_length(r) for r in routes)
        # Convert routes back to permutation
        new_perm = []
        for r in routes:
            new_perm.extend(r[1:-1])
        return new_perm, routes, max_len
    
    for gen in range(max_gen):
        # Selection
        idx1 = random.sample(range(pop_size), 2)
        idx2 = random.sample(range(pop_size), 2)
        parent1 = population[idx1[0]] if population[idx1[0]][0] <= population[idx1[1]][0] else population[idx1[1]]
        parent2 = population[idx2[0]] if population[idx2[0]][0] <= population[idx2[1]][0] else population[idx2[1]]
        p1 = parent1[1]
        p2 = parent2[1]
        # OX crossover
        n_cust = len(customers)
        a = random.randint(0, n_cust-1)
        b = random.randint(0, n_cust-1)
        if a > b:
            a, b = b, a
        child = [None] * n_cust
        child[a:b+1] = p1[a:b+1]
        pos = (b+1) % n_cust
        for gene in p2:
            if gene not in child:
                child[pos] = gene
                pos = (pos + 1) % n_cust
        remaining = [c for c in customers if c not in child]
        for i in range(n_cust):
            if child[i] is None:
                child[i] = remaining.pop()
        # Adaptive mutation
        prob = mutation_prob_start + (mutation_prob_end - mutation_prob_start) * (gen / max_gen)
        if random.random() < prob:
            i = random.randint(0, n_cust-1)
            j = random.randint(0, n_cust-1)
            child[i], child[j] = child[j], child[i]
        # Local search
        child_perm, child_routes, child_max = local_search(child)
        report_best_vrp(child_routes)
        # Replacement
        if child_max < population[-1][0]:
            population[-1] = (child_max, child_perm)
            population.sort(key=lambda x: x[0])
    
    return best_routes