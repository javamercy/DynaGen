import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    m = len(customers)
    if truck_count >= m:
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
        maxd = max(route_distance(r) for r in routes)
        if maxd < best_max - 1e-12:
            best_max = maxd
            best_routes = [list(r) for r in routes]

    # Precompute segment distance from i to j in permutation (with depot ends)
    def segment_dist(p, i, j):
        d = distance_matrix[0][p[i]] + distance_matrix[p[j]][0]
        for k in range(i, j):
            d += distance_matrix[p[k]][p[k+1]]
        return d

    # DP to find best split of permutation p into exactly truck_count routes
    def best_split(p):
        m = len(p)
        if m == 0:
            return None, None
        dp = [[float('inf')]*m for _ in range(truck_count+1)]
        split = [[-1]*m for _ in range(truck_count+1)]
        for i in range(m):
            dp[1][i] = segment_dist(p, i, m-1)
            split[1][i] = m-1
        for t in range(2, truck_count+1):
            for i in range(m - t + 1):
                best = float('inf')
                best_k = -1
                for k in range(i, m - (t-1)):
                    val = max(segment_dist(p, i, k), dp[t-1][k+1])
                    if val < best:
                        best = val
                        best_k = k
                dp[t][i] = best
                split[t][i] = best_k
        max_dist = dp[truck_count][0]
        if max_dist == float('inf'):
            return None, None
        # Reconstruct
        routes = []
        i = 0
        t_left = truck_count
        while t_left > 0:
            k = split[t_left][i]
            seg_custs = p[i:k+1]
            routes.append(seg_custs)
            i = k+1
            t_left -= 1
        full_routes = [[0] + seg + [0] for seg in routes]
        while len(full_routes) < truck_count:
            full_routes.append([0,0])
        return max_dist, full_routes

    # Genetic algorithm
    pop_size = 20
    generations = 100
    mutation_rate = 0.1

    pop = []
    for _ in range(pop_size):
        perm = customers[:]
        random.shuffle(perm)
        pop.append(perm)

    def evaluate(p):
        maxd, rts = best_split(p)
        if rts is not None:
            report_best_vrp(rts)
            return maxd
        return float('inf')

    scores = [evaluate(p) for p in pop]

    # Tournament selection with tie-breaking by index
    def select_parent():
        idx1 = random.randrange(pop_size)
        idx2 = random.randrange(pop_size)
        if scores[idx1] < scores[idx2] or (scores[idx1] == scores[idx2] and idx1 < idx2):
            return pop[idx1]
        else:
            return pop[idx2]

    def crossover(p1, p2):
        n = len(p1)
        a = random.randrange(n)
        b = random.randrange(a+1, n+1)
        child = [-1]*n
        child[a:b] = p1[a:b]
        pos = b % n
        for gene in p2:
            if gene not in child:
                child[pos] = gene
                pos = (pos+1) % n
        return child

    def mutate(perm):
        n = len(perm)
        i = random.randrange(n)
        j = random.randrange(n)
        if i != j:
            perm[i], perm[j] = perm[j], perm[i]
        return perm

    for gen in range(generations):
        new_pop = []
        sorted_idx = sorted(range(pop_size), key=lambda i: (scores[i], i))
        new_pop.append(pop[sorted_idx[0]])
        new_pop.append(pop[sorted_idx[1]])
        while len(new_pop) < pop_size:
            p1 = select_parent()
            p2 = select_parent()
            child = crossover(p1, p2)
            if random.random() < mutation_rate:
                child = mutate(child)
            new_pop.append(child)
        pop = new_pop
        scores = [evaluate(p) for p in pop]

    if best_routes is None:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0,0])
        return routes
    while len(best_routes) < truck_count:
        best_routes.append([0,0])
    return best_routes