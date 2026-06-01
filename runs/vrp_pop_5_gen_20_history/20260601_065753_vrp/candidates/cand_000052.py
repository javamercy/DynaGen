import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))

    def compute_route_length(route):
        if len(route) <= 1:
            return 0.0
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
                    new_len = lengths[r] - distance_matrix[route[p-1], route[p]] + distance_matrix[route[p-1], cust] + distance_matrix[cust, route[p]]
                    new_max = new_len
                    for rr in range(truck_count):
                        if rr != r and lengths[rr] > new_max:
                            new_max = lengths[rr]
                    if new_max < best_max:
                        best_max = new_max
                        best_r = r
                        best_p = p
                    elif new_max == best_max:
                        if (r < best_r) or (r == best_r and p < best_p):
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

    pop_size = min(50, n)
    max_gen = 5 * n
    mutation_initial = 0.3
    mutation_final = 0.05
    local_search_iterations = max(10, n // 5)

    population = []
    best_max = float('inf')
    best_routes = None

    # Greedy initial individual
    routes_constr = [[0, 0] for _ in range(truck_count)]
    lengths_constr = [0.0] * truck_count
    unassigned = set(customers)
    while unassigned:
        best_inc = float('inf')
        best_cust = -1
        best_t = -1
        best_p = -1
        for cust in list(unassigned):
            for t in range(truck_count):
                route = routes_constr[t]
                for p in range(1, len(route)):
                    new_len = lengths_constr[t] - distance_matrix[route[p-1], route[p]] + distance_matrix[route[p-1], cust] + distance_matrix[cust, route[p]]
                    new_max = new_len
                    for rr in range(truck_count):
                        if rr != t and lengths_constr[rr] > new_max:
                            new_max = lengths_constr[rr]
                    if new_max < best_inc:
                        best_inc = new_max
                        best_cust = cust
                        best_t = t
                        best_p = p
                    elif new_max == best_inc:
                        if cust < best_cust:
                            best_inc = new_max
                            best_cust = cust
                            best_t = t
                            best_p = p
        routes_constr[best_t].insert(best_p, best_cust)
        lengths_constr[best_t] = compute_route_length(routes_constr[best_t])
        unassigned.remove(best_cust)
    perm_constr = []
    for r in routes_constr:
        for node in r:
            if node != 0:
                perm_constr.append(node)
    routes_dec, max_len = decode(perm_constr)
    population.append((max_len, perm_constr))
    report_best_vrp(routes_dec)

    # Random individuals
    for _ in range(pop_size - 1):
        perm = customers[:]
        random.shuffle(perm)
        routes, max_len = decode(perm)
        population.append((max_len, perm))
        report_best_vrp(routes)
    population.sort(key=lambda x: x[0])

    def local_search(perm):
        routes, _ = decode(perm)
        improved = True
        iters = 0
        while improved and iters < local_search_iterations:
            improved = False
            for t in range(truck_count):
                route = routes[t]
                if len(route) <= 3:
                    continue
                best_route = route[:]
                best_dist = compute_route_length(route)
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = compute_route_length(new_route)
                        if new_dist < best_dist:
                            best_dist = new_dist
                            best_route = new_route
                if best_dist < compute_route_length(route):
                    routes[t] = best_route
                    improved = True
            iters += 1
        new_perm = []
        for r in routes:
            for node in r:
                if node != 0:
                    new_perm.append(node)
        return new_perm

    def mutate(perm, prob):
        if random.random() < prob:
            i = random.randint(0, len(perm)-1)
            j = random.randint(0, len(perm)-1)
            perm = perm[:]
            perm[i], perm[j] = perm[j], perm[i]
        return perm

    for gen in range(max_gen):
        mut_prob = mutation_initial - (mutation_initial - mutation_final) * (gen / max_gen)
        idx1 = random.sample(range(pop_size), 2)
        p1 = population[idx1[0]][1] if population[idx1[0]][0] <= population[idx1[1]][0] else population[idx1[1]][1]
        idx2 = random.sample(range(pop_size), 2)
        p2 = population[idx2[0]][1] if population[idx2[0]][0] <= population[idx2[1]][0] else population[idx2[1]][1]

        n_cust = len(customers)
        a = random.randint(0, n_cust-1)
        b = random.randint(0, n_cust-1)
        if a > b:
            a, b = b, a
        child = [None] * n_cust
        child[a:b+1] = p1[a:b+1]
        mapping = {}
        for i in range(a, b+1):
            mapping[p1[i]] = p2[i]
        for i in range(n_cust):
            if i < a or i > b:
                val = p2[i]
                while val in child:
                    val = mapping[val]
                child[i] = val
        used = set(child)
        remaining = [c for c in customers if c not in used]
        for i in range(n_cust):
            if child[i] is None:
                child[i] = remaining.pop()

        child = mutate(child, mut_prob)
        child = local_search(child)

        routes_child, max_child = decode(child)
        report_best_vrp(routes_child)

        if max_child < population[-1][0]:
            population[-1] = (max_child, child)
            population.sort(key=lambda x: x[0])

    return best_routes