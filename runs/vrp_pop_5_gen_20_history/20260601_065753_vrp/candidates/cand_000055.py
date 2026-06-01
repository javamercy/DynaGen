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
        return routes, lengths
    
    def local_search(routes, lengths):
        improved = True
        iterations = 0
        max_iter_local = 10 * (n + truck_count)
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
    
    pop_size = min(30, n)
    max_gen = 5 * n
    
    best_max = float('inf')
    best_routes = None
    
    # Initialize population with random permutations
    population = []
    for _ in range(pop_size):
        perm = customers[:]
        random.shuffle(perm)
        routes, lengths = decode(perm)
        routes, lengths = local_search(routes, lengths)
        max_len = max(lengths)
        population.append((max_len, perm))
        report_best_vrp(routes)
    population.sort(key=lambda x: x[0])
    
    for gen in range(1, max_gen + 1):
        # Tournament selection
        idx1 = random.sample(range(pop_size), 2)
        idx2 = random.sample(range(pop_size), 2)
        parent1 = population[idx1[0]] if population[idx1[0]][0] <= population[idx1[1]][0] else population[idx1[1]]
        parent2 = population[idx2[0]] if population[idx2[0]][0] <= population[idx2[1]][0] else population[idx2[1]]
        p = parent1[1] if random.random() < 0.5 else parent2[1]
        
        n_cust = len(customers)
        child = p[:]
        # Swap mutation
        if random.random() < 0.2:
            i = random.randint(0, n_cust-1)
            j = random.randint(0, n_cust-1)
            child[i], child[j] = child[j], child[i]
        
        routes_child, lengths_child = decode(child)
        routes_child, lengths_child = local_search(routes_child, lengths_child)
        max_child = max(lengths_child)
        report_best_vrp(routes_child)
        
        # Replace worst if better
        if max_child < population[-1][0]:
            population[-1] = (max_child, child)
            population.sort(key=lambda x: x[0])
    
    return best_routes