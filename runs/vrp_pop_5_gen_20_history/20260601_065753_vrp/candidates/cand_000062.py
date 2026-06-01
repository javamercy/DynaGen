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
                    if new_max < best_max or (new_max == best_max and (r < best_r or (r == best_r and p < best_p))):
                        best_max = new_max
                        best_r = r
                        best_p = p
            routes[best_r].insert(best_p, cust)
            lengths[best_r] = compute_route_length(routes[best_r])
        max_len = max(lengths)
        return routes, max_len

    def local_search(routes, lengths):
        improved = True
        iterations = 0
        max_iter_local = 10 * (n + truck_count)
        while improved and iterations < max_iter_local:
            improved = False
            iterations += 1
            # 2-opt for each route
            for r in range(truck_count):
                route = routes[r]
                if len(route) <= 3:
                    continue
                best_delta = 0
                best_ij = None
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_len = compute_route_length(new_route)
                        delta = new_len - lengths[r]
                        if delta < best_delta:
                            new_max = max(lengths[r] + delta, max(lengths[:r] + lengths[r+1:], default=0))
                            if new_max < max(lengths):
                                best_delta = delta
                                best_ij = (i, j)
                if best_ij is not None:
                    i, j = best_ij
                    routes[r] = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    lengths[r] = compute_route_length(routes[r])
                    improved = True
            # Relocate: move customer from longest route to another
            if improved:
                continue
            max_len = max(lengths)
            longest_routes = [i for i, l in enumerate(lengths) if l == max_len]
            if longest_routes:
                src_r = random.choice(longest_routes)
                src_route = routes[src_r]
                if len(src_route) > 2:
                    cust_idx = random.randint(1, len(src_route)-2)
                    cust = src_route[cust_idx]
                    new_src = src_route[:cust_idx] + src_route[cust_idx+1:]
                    new_src_len = compute_route_length(new_src)
                    # choose a different route
                    tgt_r = random.randrange(truck_count)
                    while tgt_r == src_r:
                        tgt_r = random.randrange(truck_count)
                    tgt_route = routes[tgt_r]
                    # try all insertion positions in target route, pick best for max
                    best_pos = -1
                    best_new_max = float('inf')
                    for p in range(1, len(tgt_route)):
                        new_tgt = tgt_route[:p] + [cust] + tgt_route[p:]
                        new_tgt_len = compute_route_length(new_tgt)
                        candidate_max = max(new_src_len, new_tgt_len, max(lengths[:src_r] + lengths[src_r+1:tgt_r] + lengths[tgt_r+1:]))
                        if candidate_max < best_new_max:
                            best_new_max = candidate_max
                            best_pos = p
                    if best_new_max < max(lengths):
                        new_tgt = tgt_route[:best_pos] + [cust] + tgt_route[best_pos:]
                        routes[src_r] = new_src
                        routes[tgt_r] = new_tgt
                        lengths[src_r] = new_src_len
                        lengths[tgt_r] = compute_route_length(new_tgt)
                        improved = True
        return routes, lengths

    def report_best_vrp(routes):
        nonlocal best_max, best_routes
        m = max(compute_route_length(r) for r in routes)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]

    pop_size = min(50, n)
    max_gen = 5 * n
    mutation_prob = 0.1

    population = []
    best_max = float('inf')
    best_routes = None

    for _ in range(pop_size):
        perm = customers[:]
        random.shuffle(perm)
        routes, max_len = decode(perm)
        routes, lengths = local_search(routes, [compute_route_length(r) for r in routes])
        max_len = max(lengths)
        population.append((max_len, perm))
        report_best_vrp(routes)
    population.sort(key=lambda x: x[0])

    for gen in range(max_gen):
        # Binary tournament selection
        idx1 = random.sample(range(pop_size), 2)
        idx2 = random.sample(range(pop_size), 2)
        parent1 = population[idx1[0]] if population[idx1[0]][0] <= population[idx1[1]][0] else population[idx1[1]]
        parent2 = population[idx2[0]] if population[idx2[0]][0] <= population[idx2[1]][0] else population[idx2[1]]
        p1, p2 = parent1[1], parent2[1]

        # Order crossover (OX)
        n_cust = len(customers)
        a = random.randint(0, n_cust-1)
        b = random.randint(0, n_cust-1)
        if a > b:
            a, b = b, a
        child = [None] * n_cust
        child[a:b+1] = p1[a:b+1]
        pos = b+1
        for gene in p2:
            if gene not in child:
                if pos >= n_cust:
                    pos = 0
                child[pos] = gene
                pos += 1

        # Inversion mutation
        if random.random() < mutation_prob:
            i = random.randint(0, n_cust-1)
            j = random.randint(0, n_cust-1)
            if i > j:
                i, j = j, i
            child[i:j+1] = child[i:j+1][::-1]

        # Decode and local search
        routes_child, max_child = decode(child)
        routes_child, lengths_child = local_search(routes_child, [compute_route_length(r) for r in routes_child])
        max_child = max(lengths_child)
        report_best_vrp(routes_child)

        # Replace worst if better
        if max_child < population[-1][0]:
            population[-1] = (max_child, child)
            population.sort(key=lambda x: x[0])

    return best_routes