import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    m = n - 1
    customers = list(range(1, n))
    dist = distance_matrix
    
    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        return sum(dist[route[i], route[i+1]] for i in range(len(route)-1))
    
    def compute_max_dist(routes):
        return max(route_distance(r) for r in routes)
    
    def decode(perm):
        def feasible(max_dist):
            routes = []
            current_route = [0]
            for cust in perm:
                temp_route = current_route + [cust]
                d = route_distance(temp_route) if len(temp_route) > 2 else dist[0, cust] + dist[cust, 0]
                if d <= max_dist:
                    current_route = temp_route
                else:
                    current_route.append(0)
                    routes.append(current_route)
                    current_route = [0, cust]
            current_route.append(0)
            routes.append(current_route)
            if len(routes) > truck_count:
                return False, None
            while len(routes) < truck_count:
                routes.append([0,0])
            return True, routes
        low = 0.0
        high = sum(dist[0,i] + dist[i,0] for i in customers) * 2
        if high == 0:
            high = 1e6
        best_routes = None
        for _ in range(50):
            mid = (low + high) / 2
            feasible_flag, routes = feasible(mid)
            if feasible_flag:
                best_routes = routes
                high = mid
            else:
                low = mid
        if best_routes is None:
            routes = []
            for cust in customers:
                routes.append([0, cust, 0])
            while len(routes) < truck_count:
                routes.append([0,0])
            best_routes = routes
        return best_routes, compute_max_dist(best_routes)
    
    def evaluate(perm):
        routes, maxd = decode(perm)
        return maxd, routes
    
    def regret_construction():
        routes = [[0,0] for _ in range(truck_count)]
        route_distances = [0.0 for _ in range(truck_count)]
        unassigned = set(customers)
        while unassigned:
            candidates = []
            for cust in unassigned:
                costs = []
                for r in range(truck_count):
                    route = routes[r]
                    for pos in range(1, len(route)):
                        delta = dist[route[pos-1], cust] + dist[cust, route[pos]] - dist[route[pos-1], route[pos]]
                        new_dist = route_distances[r] + delta
                        costs.append((new_dist, r, pos))
                costs.sort(key=lambda x: x[0])
                best = costs[0]
                second = costs[1] if len(costs) > 1 else (float('inf'), -1, -1)
                regret = second[0] - best[0]
                candidates.append((regret, cust, best[1], best[2], best[0]))
            candidates.sort(key=lambda x: (-x[0], x[1]))
            regret, cust, r_idx, pos, new_dist = candidates[0]
            routes[r_idx].insert(pos, cust)
            route_distances[r_idx] = new_dist
            unassigned.remove(cust)
        perm = []
        for r in routes:
            perm.extend(r[1:-1])
        return perm
    
    def nn_perm():
        visited = set()
        perm = []
        current = 0
        while len(visited) < m:
            best = None
            best_d = float('inf')
            for c in customers:
                if c not in visited and dist[current, c] < best_d:
                    best_d = dist[current, c]
                    best = c
            if best is None:
                break
            visited.add(best)
            perm.append(best)
            current = best
        for c in customers:
            if c not in visited:
                perm.append(c)
        return perm
    
    pop_size = min(20, max(m, 10))
    population = []
    try:
        perm_regret = regret_construction()
        max_regret, routes_regret = evaluate(perm_regret)
        population.append((max_regret, perm_regret, routes_regret))
    except:
        pass
    try:
        perm_nn = nn_perm()
        max_nn, routes_nn = evaluate(perm_nn)
        population.append((max_nn, perm_nn, routes_nn))
    except:
        pass
    while len(population) < pop_size:
        perm = customers[:]
        random.shuffle(perm)
        maxd, routes = evaluate(perm)
        population.append((maxd, perm, routes))
    population.sort(key=lambda x: x[0])
    best_fitness = population[0][0]
    best_solution = population[0][2]
    try:
        report_best_vrp(best_solution)
    except:
        pass
    max_generations = 500
    for gen in range(max_generations):
        def tournament():
            idx1 = random.randint(0, pop_size-1)
            idx2 = random.randint(0, pop_size-1)
            idx3 = random.randint(0, pop_size-1)
            candidates = [population[idx1], population[idx2], population[idx3]]
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        parent1 = tournament()
        parent2 = tournament()
        def order_crossover(p1, p2):
            size = len(p1)
            a = random.randint(0, size-1)
            b = random.randint(0, size-1)
            if a > b:
                a, b = b, a
            child = [None]*size
            child[a:b+1] = p1[a:b+1]
            pos = (b+1) % size
            for g in p2:
                if g not in child:
                    child[pos] = g
                    pos = (pos+1) % size
            return child
        child_perm = order_crossover(parent1, parent2)
        if random.random() < 0.2:
            i, j = random.sample(range(m), 2)
            child_perm[i], child_perm[j] = child_perm[j], child_perm[i]
        if random.random() < 0.2:
            i = random.randint(0, m-1)
            cust = child_perm.pop(i)
            j = random.randint(0, m-1)
            child_perm.insert(j, cust)
        child_fitness, child_routes = evaluate(child_perm)
        worst = max(population, key=lambda x: x[0])
        if child_fitness < worst[0]:
            population.remove(worst)
            population.append((child_fitness, child_perm, child_routes))
            population.sort(key=lambda x: x[0])
            if child_fitness < best_fitness:
                best_fitness = child_fitness
                best_solution = child_routes
                try:
                    report_best_vrp(best_solution)
                except:
                    pass
    return best_solution