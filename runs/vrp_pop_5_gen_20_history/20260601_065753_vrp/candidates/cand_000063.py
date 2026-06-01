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
        max_iter_local = 20 * (n + truck_count)
        while improved and iterations < max_iter_local:
            improved = False
            iterations += 1
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
        return routes, lengths
    
    def report_best_vrp(routes):
        nonlocal best_max, best_routes
        m = max(compute_route_length(r) for r in routes)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]
    
    pop_size = min(100, n)
    max_gen = 5 * n
    stagnation_limit = max_gen // 5
    
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
    
    no_improve_gen = 0
    for gen in range(1, max_gen + 1):
        mutation_prob = 0.3 * (1 - gen / max_gen)
        # Roulette wheel selection (fitness proportional to 1/(1+max_len))
        fitness = [1.0 / (1.0 + pop[i][0]) for i in range(pop_size)]
        total_fitness = sum(fitness)
        prob = [f / total_fitness for f in fitness]
        cum_prob = [0.0] * (pop_size + 1)
        for i in range(pop_size):
            cum_prob[i+1] = cum_prob[i] + prob[i]
        r1 = random.random()
        for i in range(pop_size):
            if cum_prob[i] <= r1 < cum_prob[i+1]:
                parent1 = population[i]
                break
        r2 = random.random()
        for i in range(pop_size):
            if cum_prob[i] <= r2 < cum_prob[i+1]:
                parent2 = population[i]
                break
        p1, p2 = parent1[1], parent2[1]
        
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
        if len(used) != n_cust:
            remaining = [c for c in customers if c not in used]
            for i in range(n_cust):
                if child[i] is None:
                    child[i] = remaining.pop()
        
        if random.random() < mutation_prob:
            i = random.randint(0, n_cust-1)
            j = random.randint(0, n_cust-1)
            child[i], child[j] = child[j], child[i]
        
        routes_child, max_child = decode(child)
        routes_child, lengths_child = local_search(routes_child, [compute_route_length(r) for r in routes_child])
        max_child = max(lengths_child)
        report_best_vrp(routes_child)
        
        if max_child < population[-1][0]:
            population[-1] = (max_child, child)
            population.sort(key=lambda x: x[0])
            if max_child < best_max:
                no_improve_gen = 0
            else:
                no_improve_gen += 1
        else:
            no_improve_gen += 1
        
        if no_improve_gen >= stagnation_limit:
            no_improve_gen = 0
            for i in range(pop_size // 2, pop_size):
                perm = customers[:]
                random.shuffle(perm)
                routes, _ = decode(perm)
                routes, lengths = local_search(routes, [compute_route_length(r) for r in routes])
                max_len = max(lengths)
                population[i] = (max_len, perm)
                report_best_vrp(routes)
            population.sort(key=lambda x: x[0])
    
    return best_routes