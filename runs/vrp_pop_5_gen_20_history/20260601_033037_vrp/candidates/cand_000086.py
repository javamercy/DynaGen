import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    L = n - 1  # number of customers

    # trivial cases
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    if truck_count >= L:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    # set random seed for reproducibility
    random.seed(0)

    # global best
    best_routes = None
    best_max = float('inf')

    def route_distance(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        new_max = max(route_distance(r) for r in routes)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [list(r) for r in routes]

    # min-max insertion construction (from parents) to seed a good solution
    def construct_minmax():
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = set(customers)
        while unassigned:
            best_cust = None
            best_route_idx = None
            best_pos = None
            best_new_max = float('inf')
            sorted_cust = sorted(unassigned)
            for cust in sorted_cust:
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        temp_routes = [list(r) for r in routes]
                        temp_routes[r_idx] = new_route
                        new_max = max(route_distance(r) for r in temp_routes)
                        if new_max < best_new_max - 1e-12:
                            best_new_max = new_max
                            best_cust = cust
                            best_route_idx = r_idx
                            best_pos = pos
                        elif abs(new_max - best_new_max) < 1e-12:
                            if cust < best_cust or (cust == best_cust and r_idx < best_route_idx) or \
                               (cust == best_cust and r_idx == best_route_idx and pos < best_pos):
                                best_new_max = new_max
                                best_cust = cust
                                best_route_idx = r_idx
                                best_pos = pos
            routes[best_route_idx].insert(best_pos, best_cust)
            unassigned.remove(best_cust)
        return routes

    # decode a permutation into routes using DP to minimize max route distance
    def decode(perm):
        # perm: list of customers in order
        # precompute segment cost matrix
        Lp = len(perm)
        cost = [[0.0]*Lp for _ in range(Lp)]
        for i in range(Lp):
            for j in range(i, Lp):
                dist = distance_matrix[0, perm[i]]
                for k in range(i, j):
                    dist += distance_matrix[perm[k], perm[k+1]]
                dist += distance_matrix[perm[j], 0]
                cost[i][j] = dist

        # DP: dp[k][j] = (min_max, prev_i) for first j customers (0..j-1)
        INF = 1e100
        dp = [[(INF, -1) for _ in range(Lp+1)] for _ in range(truck_count+1)]
        dp[0][0] = (0.0, -1)
        # k: number of routes used
        for k in range(1, truck_count+1):
            for j in range(0, Lp+1):
                best = INF
                best_i = -1
                # try all possible i where previous route ended at i-1 (i customers in previous routes)
                for i in range(0, j+1):
                    prev_max, _ = dp[k-1][i]
                    if prev_max >= INF:
                        continue
                    if i == j:  # empty route
                        seg_cost = 0.0
                    else:
                        seg_cost = cost[i][j-1]
                    cand = max(prev_max, seg_cost)
                    if cand < best - 1e-12:
                        best = cand
                        best_i = i
                    elif abs(cand - best) < 1e-12:
                        # tie-break: prefer smaller i (earlier cut)
                        if i < best_i:
                            best_i = i
                dp[k][j] = (best, best_i)

        # reconstruct routes from dp
        routes = []
        j = Lp
        k = truck_count
        while k > 0:
            _, i = dp[k][j]
            if i == j:
                routes.append([0, 0])
            else:
                seg = perm[i:j]
                routes.append([0] + seg + [0])
            j = i
            k -= 1
        routes.reverse()
        return routes

    # initial population
    pop_size = min(50, L * 2)
    pop = []
    # seed with minmax solution
    minmax_routes = construct_minmax()
    # convert minmax solutions to permutation
    perm_seed = []
    for r in minmax_routes:
        if len(r) > 2:
            perm_seed.extend(r[1:-1])
    # if the solution uses all customers, it should be a permutation; else random fill
    if len(set(perm_seed)) == L:
        pop.append(perm_seed)
    else:
        # generate random permutation
        perm = customers[:]
        random.shuffle(perm)
        pop.append(perm)
    # fill rest with random permutations
    while len(pop) < pop_size:
        perm = customers[:]
        random.shuffle(perm)
        pop.append(perm)

    # evaluate initial population
    fitness = []
    for perm in pop:
        routes = decode(perm)
        report_best_vrp(routes)
        max_dist = max(route_distance(r) for r in routes)
        fitness.append(max_dist)

    # genetic algorithm parameters
    gen_limit = min(200, L * 2)
    elite_size = max(1, pop_size // 10)
    tournament_size = 3
    crossover_rate = 0.9
    mutation_rate = 0.1

    for gen in range(gen_limit):
        # create next population
        new_pop = []
        # elitism
        sorted_indices = sorted(range(pop_size), key=lambda i: (fitness[i], i))
        for idx in sorted_indices[:elite_size]:
            new_pop.append(pop[idx])
        while len(new_pop) < pop_size:
            # tournament selection
            def tournament():
                best_idx = None
                best_fit = INF
                for _ in range(tournament_size):
                    idx = random.randrange(pop_size)
                    if fitness[idx] < best_fit - 1e-12:
                        best_fit = fitness[idx]
                        best_idx = idx
                    elif abs(fitness[idx] - best_fit) < 1e-12:
                        if idx < best_idx:
                            best_idx = idx
                return pop[best_idx]
            parent1 = tournament()
            parent2 = tournament()
            # crossover
            if random.random() < crossover_rate:
                child = order_crossover(parent1, parent2)
            else:
                child = parent1[:]
            # mutation
            if random.random() < mutation_rate:
                child = swap_mutation(child)
            new_pop.append(child)
        pop = new_pop
        # evaluate offspring
        fitness = []
        for perm in pop:
            routes = decode(perm)
            report_best_vrp(routes)
            max_dist = max(route_distance(r) for r in routes)
            fitness.append(max_dist)

    return best_routes if best_routes is not None else decode(pop[0])

def order_crossover(p1, p2):
    size = len(p1)
    a = random.randrange(size)
    b = random.randrange(size)
    if a > b:
        a, b = b, a
    child = [None]*size
    child[a:b+1] = p1[a:b+1]
    remaining = [x for x in p2 if x not in child]
    idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = remaining[idx]
            idx += 1
    return child

def swap_mutation(perm):
    size = len(perm)
    if size < 2:
        return perm
    i = random.randrange(size)
    j = random.randrange(size)
    perm = perm[:]
    perm[i], perm[j] = perm[j], perm[i]
    return perm