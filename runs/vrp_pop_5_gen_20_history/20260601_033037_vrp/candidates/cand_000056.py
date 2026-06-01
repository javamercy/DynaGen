import numpy as np
import random
import heapq

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    if truck_count >= n - 1:
        customers = list(range(1, n))
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_distance(route):
        total = 0.0
        for i in range(len(route) - 1):
            total += distance_matrix[route[i], route[i + 1]]
        return total

    best_routes = None
    best_max = float('inf')

    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        new_max = max(route_distance(r) for r in routes)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [list(r) for r in routes]

    random.seed(0)

    # Decoding: permutation to routes via DP minimizing max route distance
    def decode(perm):
        m = len(perm)
        if m == 0:
            return [[0, 0] for _ in range(truck_count)], 0.0
        seq_dist = [distance_matrix[perm[i], perm[i + 1]] for i in range(m - 1)]
        depot_first = [distance_matrix[0, perm[i]] for i in range(m)]
        last_depot = [distance_matrix[perm[i], 0] for i in range(m)]
        INF = 1e100
        dp = [[INF] * (truck_count + 1) for _ in range(m + 1)]
        choice = [[-1] * (truck_count + 1) for _ in range(m + 1)]
        dp[0][0] = 0.0
        for i in range(1, m + 1):
            for r in range(1, truck_count + 1):
                for j in range(0, i):
                    if dp[j][r - 1] == INF:
                        continue
                    seg = depot_first[j] + (sum(seq_dist[j:i - 1]) if j < i - 1 else 0) + last_depot[i - 1]
                    new_max = max(dp[j][r - 1], seg)
                    if new_max < dp[i][r]:
                        dp[i][r] = new_max
                        choice[i][r] = j
        routes = []
        i = m
        r = truck_count
        while r > 0:
            j = choice[i][r]
            seg_cust = perm[j:i]
            route = [0] + seg_cust + [0]
            routes.append(route)
            i = j
            r -= 1
        routes.reverse()
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes, dp[m][truck_count]

    # Initial solution from deterministic savings
    def savings_initial():
        customers_local = list(range(1, n))
        route_list = [[0, c, 0] for c in customers_local]
        savings = []
        for i in customers_local:
            for j in customers_local:
                if i < j:
                    s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
                    savings.append((-s, i, j))
        heapq.heapify(savings)
        cust_to_route = {c: idx for idx, c in enumerate(customers_local)}
        endpoints = [(c, c) for c in customers_local]
        while len(route_list) > truck_count and savings:
            neg_s, i, j = heapq.heappop(savings)
            if i not in cust_to_route or j not in cust_to_route:
                continue
            ri = cust_to_route[i]
            rj = cust_to_route[j]
            if ri == rj:
                continue
            first_i, last_i = endpoints[ri]
            first_j, last_j = endpoints[rj]
            merged = None
            if i == last_i and j == first_j:
                merged = route_list[ri][:-1] + route_list[rj][1:]
            elif j == last_j and i == first_i:
                merged = route_list[rj][:-1] + route_list[ri][1:]
            elif i == first_i and j == last_j:
                merged = route_list[rj][:-1] + route_list[ri][1:]
            elif j == first_j and i == last_i:
                merged = route_list[ri][:-1] + route_list[rj][1:]
            else:
                continue
            new_route_list = [r for idx, r in enumerate(route_list) if idx not in (ri, rj)]
            new_route_list.append(merged)
            route_list = new_route_list
            cust_to_route.clear()
            endpoints = []
            for idx2, r in enumerate(route_list):
                interior = r[1:-1]
                for c in interior:
                    cust_to_route[c] = idx2
                first_c = interior[0] if interior else None
                last_c = interior[-1] if interior else None
                endpoints.append((first_c, last_c))
        while len(route_list) > truck_count:
            dists = [(route_distance(r), idx) for idx, r in enumerate(route_list)]
            dists.sort(key=lambda x: (x[0], x[1]))
            idx1 = dists[0][1]
            idx2 = dists[1][1]
            merged = route_list[idx1][:-1] + route_list[idx2][1:]
            route_list = [r for i, r in enumerate(route_list) if i not in (idx1, idx2)]
            route_list.append(merged)
        return route_list

    base_routes = savings_initial()
    base_perm = []
    for r in base_routes:
        base_perm.extend(r[1:-1])

    pop_size = 10
    pop = [base_perm[:]]
    while len(pop) < pop_size:
        perm = base_perm[:]
        num_swaps = random.randint(1, 3)
        for _ in range(num_swaps):
            a = random.randint(0, len(perm) - 1)
            b = random.randint(0, len(perm) - 1)
            perm[a], perm[b] = perm[b], perm[a]
        pop.append(perm)

    def evaluate(perm):
        routes, max_dist = decode(perm)
        return max_dist, routes

    fits_routes = [evaluate(perm) for perm in pop]
    fits = [f[0] for f in fits_routes]
    for perm in pop:
        _, routes = evaluate(perm)
        report_best_vrp(routes)

    generations = 20
    for gen in range(generations):
        new_pop = []
        sorted_idx = sorted(range(len(pop)), key=lambda i: fits[i])
        new_pop.append(pop[sorted_idx[0]])
        new_pop.append(pop[sorted_idx[1]])
        _, routes_best = evaluate(pop[sorted_idx[0]])
        report_best_vrp(routes_best)
        while len(new_pop) < pop_size:
            idx1, idx2 = random.sample(range(len(pop)), 2)
            if fits[idx1] < fits[idx2]:
                parent1 = pop[idx1]
            else:
                parent1 = pop[idx2]
            idx3, idx4 = random.sample(range(len(pop)), 2)
            if fits[idx3] < fits[idx4]:
                parent2 = pop[idx3]
            else:
                parent2 = pop[idx4]
            if random.random() < 0.8:
                a = random.randint(0, len(parent1) - 1)
                b = random.randint(a, len(parent1) - 1)
                child = [None] * len(parent1)
                child[a:b + 1] = parent1[a:b + 1]
                cur = (b + 1) % len(parent1)
                for c in parent2:
                    if c not in child:
                        child[cur] = c
                        cur = (cur + 1) % len(parent1)
            else:
                child = parent1[:]
            if random.random() < 0.2:
                i = random.randint(0, len(child) - 1)
                j = random.randint(0, len(child) - 1)
                child[i], child[j] = child[j], child[i]
            new_pop.append(child)
        pop = new_pop
        fits = [evaluate(perm)[0] for perm in pop]

    if best_routes is None:
        _, final_routes = evaluate(pop[0])
        return final_routes
    return best_routes