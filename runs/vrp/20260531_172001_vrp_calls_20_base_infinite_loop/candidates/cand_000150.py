import numpy as np
import random
from math import exp

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]

    def route_length(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    def max_route_len(routes):
        return max(route_length(r) for r in routes) if routes else float('inf')

    # DP split: given permutation (list of customer nodes), return a list of routes (each [0,...,0])
    # that minimizes max route length given truck_count.
    def split_permutation(perm):
        cust_count = len(perm)
        if cust_count == 0:
            return [[0,0] for _ in range(truck_count)]
        # upper bound for binary search: total distance of all customers in one route
        dist_all = distance_matrix[0, perm[0]] + distance_matrix[perm[-1], 0]
        for i in range(cust_count-1):
            dist_all += distance_matrix[perm[i], perm[i+1]]
        lo = 0.0
        hi = dist_all
        best_routes = None
        for _ in range(50):  # binary search iterations
            mid = (lo + hi) / 2
            # DP: check if feasible with at most truck_count routes
            # dp[i] = min routes for prefix ending at i (inclusive)
            # Use large number
            INF = 10**9
            dp = [INF] * (cust_count + 1)
            dp[0] = 0
            # Precompute route lengths for subsegments? Not necessary, compute on fly
            for i in range(1, cust_count+1):
                # consider all j < i
                for j in range(i):
                    # route from perm[j] to perm[i-1], with depot at ends
                    route = [0] + list(perm[j:i]) + [0]
                    l = route_length(route)
                    if l <= mid + 1e-9:
                        if dp[j] + 1 < dp[i]:
                            dp[i] = dp[j] + 1
            if dp[cust_count] <= truck_count:
                best_routes = None
                # reconstruct
                routes_for_split = []
                remaining = cust_count
                while remaining > 0:
                    for j in range(remaining-1, -1, -1):
                        route = [0] + list(perm[j:remaining]) + [0]
                        if route_length(route) <= mid + 1e-9:
                            if dp[remaining] == dp[j] + 1:
                                routes_for_split.insert(0, route)
                                remaining = j
                                break
                # fill with empty routes
                while len(routes_for_split) < truck_count:
                    routes_for_split.append([0,0])
                best_routes = routes_for_split
                hi = mid
            else:
                lo = mid
        if best_routes is None:
            # fallback: assign all to one route, rest empty
            routes = [[0] + list(perm) + [0]]
            routes += [[0,0] for _ in range(truck_count-1)]
            return routes
        else:
            return best_routes

    def routes_to_permutation(routes):
        perm = []
        for r in routes:
            for node in r:
                if node != 0:
                    perm.append(node)
        return perm

    def make_initial_solution():
        # greedy construction from parents (min-max with regret)
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = set(range(1, n))
        while unassigned:
            candidates = []
            for cust in unassigned:
                best_insert = None
                best_second = None
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        new_len = route_length(route) + cost
                        other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                        new_max = max(new_len, *other_lens)
                        if best_insert is None or new_max < best_insert[0] or (new_max == best_insert[0] and cost < best_insert[1]):
                            best_second = best_insert
                            best_insert = (new_max, cost, r_idx, pos)
                        else:
                            if best_second is None or new_max < best_second[0] or (new_max == best_second[0] and cost < best_second[1]):
                                best_second = (new_max, cost, r_idx, pos)
                if best_insert is None:
                    continue
                second_val = best_second[0] if best_second is not None else (best_insert[0] + 1e9)
                regret = second_val - best_insert[0]
                candidates.append((best_insert[0], regret, best_insert[1], best_insert[2], best_insert[3], cust))
            if not candidates:
                break
            candidates.sort(key=lambda x: (x[0], -x[1], -x[2], x[5]))
            _, _, _, r_idx, pos, cust = candidates[0]
            routes[r_idx].insert(pos, cust)
            unassigned.remove(cust)
        return routes

    # Genetic Algorithm
    pop_size = min(20, n * truck_count)
    if pop_size < 4:
        pop_size = 4
    max_generations = min(30, n * 2)
    crossover_rate = 0.8
    mutation_rate = 0.1
    tourn_size = 3

    # initial population: mix of greedy and random
    population = []
    # greedy solutions
    for _ in range(pop_size // 2):
        routes = make_initial_solution()
        perm = routes_to_permutation(routes)
        population.append(perm)
    # random permutations
    for _ in range(pop_size - len(population)):
        perm = list(range(1, n))
        random.shuffle(perm)
        population.append(perm)

    # evaluate fitness
    def fitness(perm):
        routes = split_permutation(perm)
        return max_route_len(routes)

    best_perm = None
    best_fit = float('inf')
    for idx, perm in enumerate(population):
        routes = split_permutation(perm)
        fit = max_route_len(routes)
        if fit < best_fit - 1e-12:
            best_fit = fit
            best_perm = perm[:]
            report_best_vrp(routes)

    def crossover(p1, p2):
        # order crossover (OX)
        n = len(p1)
        a = random.randint(0, n-1)
        b = random.randint(a, n-1)
        child = [0] * n
        child[a:b+1] = p1[a:b+1]
        remaining = [x for x in p2 if x not in child[a:b+1]]
        pos = 0
        for i in range(n):
            if child[i] == 0:
                child[i] = remaining[pos]
                pos += 1
        return child

    def mutate(perm):
        # swap two random positions
        i, j = random.sample(range(len(perm)), 2)
        perm[i], perm[j] = perm[j], perm[i]
        return perm

    # local search on routes (only improvement)
    def local_search(routes):
        improved = True
        while improved:
            improved = False
            # inter_relocate
            lengths = [route_length(r) for r in routes]
            max_idx = int(np.argmax(lengths))
            max_route = routes[max_idx]
            if len(max_route) > 2:
                best_delta = 0.0
                best_move = None
                for cust in max_route[1:-1]:
                    new_max_route = [x for x in max_route if x != cust]
                    new_max_len = route_length(new_max_route)
                    for r_idx in range(truck_count):
                        if r_idx == max_idx:
                            continue
                        other_route = routes[r_idx]
                        for pos in range(1, len(other_route)):
                            new_other = other_route[:pos] + [cust] + other_route[pos:]
                            new_other_len = route_length(new_other)
                            other_lens = [lengths[i] for i in range(truck_count) if i not in (max_idx, r_idx)]
                            new_max_candidate = max(new_max_len, new_other_len, *other_lens)
                            if new_max_candidate < current_max - 1e-12:
                                delta = current_max - new_max_candidate
                                if delta > best_delta:
                                    best_delta = delta
                                    best_move = (cust, max_idx, r_idx, pos, new_max_candidate)
                if best_move:
                    cust, from_idx, to_idx, pos, new_max_val = best_move
                    routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                    routes[to_idx].insert(pos, cust)
                    current_max = new_max_val
                    improved = True
            # inter_swap
            lengths = [route_length(r) for r in routes]
            max_idx = int(np.argmax(lengths))
            max_route = routes[max_idx]
            if len(max_route) > 2:
                best_delta = 0.0
                best_move = None
                for cust_i in max_route[1:-1]:
                    for other_idx in range(truck_count):
                        if other_idx == max_idx:
                            continue
                        other_route = routes[other_idx]
                        for cust_j in other_route[1:-1]:
                            new_max_route = [x if x != cust_i else cust_j for x in max_route]
                            new_other_route = [x if x != cust_j else cust_i for x in other_route]
                            new_max_len = route_length(new_max_route)
                            new_other_len = route_length(new_other_route)
                            other_lens = [lengths[i] for i in range(truck_count) if i not in (max_idx, other_idx)]
                            new_max_candidate = max(new_max_len, new_other_len, *other_lens)
                            if new_max_candidate < current_max - 1e-12:
                                delta = current_max - new_max_candidate
                                if delta > best_delta:
                                    best_delta = delta
                                    best_move = (cust_i, max_idx, cust_j, other_idx, new_max_candidate)
                if best_move:
                    cust_i, from_idx, cust_j, to_idx, new_max_val = best_move
                    routes[from_idx] = [x if x != cust_i else cust_j for x in routes[from_idx]]
                    routes[to_idx] = [x if x != cust_j else cust_i for x in routes[to_idx]]
                    current_max = new_max_val
                    improved = True
            # intra_2opt
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                best_delta = 0.0
                best_ij = None
                for i in range(1, len(route)-2):
                    for k in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                        new_len = route_length(new_route)
                        if new_len < route_length(route) - 1e-12:
                            delta = route_length(route) - new_len
                            if delta > best_delta:
                                best_delta = delta
                                best_ij = (i, k, r_idx)
                if best_ij:
                    i, k, r_idx = best_ij
                    routes[r_idx] = route[:i] + route[i:k+1][::-1] + route[k+1:]
                    new_max = max_route_len(routes)
                    if new_max < current_max:
                        current_max = new_max
                        improved = True
        return routes

    for gen in range(max_generations):
        # selection
        new_pop = []
        # elitism: carry over best
        if best_perm is not None:
            new_pop.append(best_perm[:])
        while len(new_pop) < pop_size:
            # tournament
            contestants = random.sample(population, tourn_size)
            best_c = min(contestants, key=lambda p: fitness(p))
            parent1 = best_c[:]
            contestants = random.sample(population, tourn_size)
            best_c = min(contestants, key=lambda p: fitness(p))
            parent2 = best_c[:]
            child = None
            if random.random() < crossover_rate:
                child = crossover(parent1, parent2)
            else:
                child = parent1[:]
            if random.random() < mutation_rate:
                child = mutate(child)
            new_pop.append(child)
        population = new_pop

        # evaluate and apply local search to best
        # find best in population
        best_in_pop = min(population, key=lambda p: fitness(p))
        if best_in_pop != best_perm and fitness(best_in_pop) < best_fit - 1e-12:
            routes = split_permutation(best_in_pop)
            fit = max_route_len(routes)
            if fit < best_fit - 1e-12:
                best_fit = fit
                best_perm = best_in_pop[:]
                report_best_vrp(routes)
        # local search on best
        routes = split_permutation(best_perm)
        routes = local_search(routes)
        new_fit = max_route_len(routes)
        if new_fit < best_fit - 1e-12:
            best_fit = new_fit
            best_perm = routes_to_permutation(routes)
            report_best_vrp(routes)

    # final solution
    if best_perm is None:
        routes = make_initial_solution()
        return routes
    else:
        routes = split_permutation(best_perm)
        return routes