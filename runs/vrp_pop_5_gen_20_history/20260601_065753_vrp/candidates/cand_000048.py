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
    
    pop_size = min(100, n)
    max_gen = 5 * n
    def mutation_prob(gen):
        return 0.25 - 0.20 * (gen / max_gen)
    
    population = []
    best_max = float('inf')
    best_routes = None
    for _ in range(pop_size):
        perm = customers[:]
        random.shuffle(perm)
        routes, max_len = decode(perm)
        population.append((max_len, perm))
        report_best_vrp(routes)
    population.sort(key=lambda x: x[0])
    
    for gen in range(max_gen):
        idx1 = random.sample(range(pop_size), 2)
        idx2 = random.sample(range(pop_size), 2)
        parent1 = population[idx1[0]] if population[idx1[0]][0] <= population[idx1[1]][0] else population[idx1[1]]
        parent2 = population[idx2[0]] if population[idx2[0]][0] <= population[idx2[1]][0] else population[idx2[1]]
        p1 = parent1[1]
        p2 = parent2[1]
        
        n_cust = len(customers)
        a = random.randint(0, n_cust-1)
        b = random.randint(0, n_cust-1)
        if a > b:
            a, b = b, a
        # Order crossover (OX)
        child = [None] * n_cust
        child[a:b+1] = p1[a:b+1]
        pos = (b + 1) % n_cust
        for i in range(n_cust):
            idx = (b + 1 + i) % n_cust
            if p2[idx] not in child:
                child[pos] = p2[idx]
                pos = (pos + 1) % n_cust
        
        if random.random() < mutation_prob(gen):
            i = random.randint(0, n_cust-1)
            j = random.randint(0, n_cust-1)
            child[i], child[j] = child[j], child[i]
        
        routes_child, max_child = decode(child)
        report_best_vrp(routes_child)
        
        if max_child < population[-1][0]:
            population[-1] = (max_child, child)
            population.sort(key=lambda x: x[0])
    
    return best_routes