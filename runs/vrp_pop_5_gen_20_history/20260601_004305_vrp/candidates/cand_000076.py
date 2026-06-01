import numpy as np
import math
import random

def solve_vrp(distance_matrix, truck_count):
    random.seed(42)
    n = distance_matrix.shape[0]
    m = n - 1
    customers = list(range(1, n))
    if truck_count >= m:
        routes = [[0, i, 0] for i in customers] + [[0, 0]] * (truck_count - m)
        report_best_vrp(routes)
        return routes

    def decode(perm):
        seg_dist = [[0] * (m + 1) for _ in range(m)]
        for l in range(m):
            acc = distance_matrix[0][perm[l]]
            for r in range(l + 1, m + 1):
                if r > l + 1:
                    acc += distance_matrix[perm[r - 2]][perm[r - 1]]
                if r == l + 1:
                    seg_dist[l][r] = distance_matrix[0][perm[l]] + distance_matrix[perm[l]][0]
                else:
                    seg_dist[l][r] = acc + distance_matrix[perm[r - 1]][0]
        dp = [[math.inf] * (truck_count + 1) for _ in range(m + 1)]
        choice = [[-1] * (truck_count + 1) for _ in range(m + 1)]
        dp[0][0] = 0
        for i in range(1, m + 1):
            for t in range(1, min(i, truck_count) + 1):
                best_val = math.inf
                best_j = -1
                for j in range(t - 1, i):
                    if dp[j][t - 1] < math.inf:
                        cand = max(dp[j][t - 1], seg_dist[j][i])
                        if cand < best_val or (cand == best_val and j < best_j):
                            best_val = cand
                            best_j = j
                dp[i][t] = best_val
                choice[i][t] = best_j
        routes = []
        i = m
        t = truck_count
        while t > 0:
            j = choice[i][t]
            seg = perm[j:i]
            routes.append([0] + seg + [0])
            i = j
            t -= 1
        routes.reverse()
        while len(routes) < truck_count:
            routes.append([0, 0])
        max_dist = 0
        for route in routes:
            d = 0
            for k in range(len(route) - 1):
                d += distance_matrix[route[k]][route[k+1]]
            if d > max_dist:
                max_dist = d
        return routes, max_dist

    pop_size = 20
    max_iter = 200 * n
    crossover_rate = 0.8
    mutation_rate = 0.1
    tourn_size = 3

    pop = []
    for _ in range(pop_size):
        perm = list(range(1, n))
        random.shuffle(perm)
        routes, obj = decode(perm)
        pop.append((perm, routes, obj))
    pop.sort(key=lambda x: (x[2], x[0]))
    best_perm, best_routes, best_obj = pop[0]
    report_best_vrp(best_routes)

    for it in range(max_iter):
        def tournament():
            best_idx = random.randrange(pop_size)
            for _ in range(tourn_size - 1):
                idx = random.randrange(pop_size)
                if pop[idx][2] < pop[best_idx][2] or (pop[idx][2] == pop[best_idx][2] and pop[idx][0] < pop[best_idx][0]):
                    best_idx = idx
            return pop[best_idx][0]
        p1 = tournament()
        p2 = tournament()
        if random.random() < crossover_rate:
            n_cust = m
            start = random.randrange(n_cust)
            end = random.randrange(n_cust)
            if start > end:
                start, end = end, start
            child = [None] * n_cust
            child[start:end+1] = p1[start:end+1]
            ptr = 0
            for i in range(n_cust):
                if child[i] is None:
                    while ptr < n_cust and p2[ptr] in child:
                        ptr += 1
                    child[i] = p2[ptr]
                    ptr += 1
        else:
            child = p1[:]
        for i in range(m):
            if random.random() < mutation_rate:
                j = random.randrange(m)
                child[i], child[j] = child[j], child[i]
        child_routes, child_obj = decode(child)
        worst_idx = pop_size - 1
        if child_obj < pop[worst_idx][2] or (child_obj == pop[worst_idx][2] and child < pop[worst_idx][0]):
            pop[worst_idx] = (child, child_routes, child_obj)
            pop.sort(key=lambda x: (x[2], x[0]))
            if child_obj < best_obj:
                best_obj = child_obj
                best_routes = child_routes
                report_best_vrp(best_routes)

    return best_routes