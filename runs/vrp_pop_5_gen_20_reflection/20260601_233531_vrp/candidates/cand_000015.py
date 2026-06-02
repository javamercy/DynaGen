import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    
    customer_count = n - 1
    customers = list(range(1, n))
    depot_dist = [distance_matrix[0, c] for c in customers]
    max_depot_dist = max(depot_dist) if depot_dist else 1.0
    if max_depot_dist == 0:
        max_depot_dist = 1.0
    
    def decode(assignment):
        truck_customers = [[] for _ in range(truck_count)]
        for i, cust in enumerate(customers):
            truck_customers[assignment[i]].append(cust)
        routes = []
        dists = []
        for tc in truck_customers:
            if not tc:
                routes.append([0, 0])
                dists.append(0.0)
            else:
                route = [0]
                unvisited = set(tc)
                current = 0
                while unvisited:
                    nearest = min(unvisited, key=lambda x: distance_matrix[current, x])
                    route.append(nearest)
                    unvisited.remove(nearest)
                    current = nearest
                route.append(0)
                routes.append(route)
                d = sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
                dists.append(d)
        # 2-opt improvement
        for idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            improved = True
            max_iters = len(route) * len(route)
            iters = 0
            while improved and iters < max_iters:
                improved = False
                iters += 1
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        old_cost = sum(distance_matrix[route[k], route[k+1]] for k in range(len(route)-1))
                        new_cost = sum(distance_matrix[new_route[k], new_route[k+1]] for k in range(len(new_route)-1))
                        if new_cost < old_cost:
                            route[:] = new_route
                            dists[idx] = new_cost
                            improved = True
                            break
                    if improved:
                        break
        max_dist = max(dists) if dists else 0.0
        total_dist = sum(dists)
        return routes, max_dist, total_dist
    
    pop_size = min(50, customer_count * 2)
    generations = min(100, customer_count * 2)
    mutation_rate = 1.0 / customer_count if customer_count > 0 else 0.1
    
    population = []
    for _ in range(pop_size):
        assignment = [random.randrange(0, truck_count) for _ in range(customer_count)]
        routes, max_dist, total_dist = decode(assignment)
        population.append((assignment, routes, max_dist, total_dist))
    
    population.sort(key=lambda x: (x[2], x[3]))
    best = population[0]
    report_best_vrp(best[1])
    
    for gen in range(generations):
        new_pop = []
        new_pop.extend(population[:2])
        while len(new_pop) < pop_size:
            i1 = random.randrange(pop_size)
            i2 = random.randrange(pop_size)
            par1 = population[i1] if population[i1][2] < population[i2][2] else population[i2]
            i3 = random.randrange(pop_size)
            i4 = random.randrange(pop_size)
            par2 = population[i3] if population[i3][2] < population[i4][2] else population[i4]
            # ensure par1 is the better (smaller max_dist)
            if par1[2] > par2[2]:
                par1, par2 = par2, par1
            child_assign = []
            for i in range(customer_count):
                d = depot_dist[i]
                p = 1.0 - d / max_depot_dist if max_depot_dist > 0 else 1.0
                if random.random() < p:
                    child_assign.append(par1[0][i])
                else:
                    child_assign.append(par2[0][i])
            for i in range(customer_count):
                if random.random() < mutation_rate:
                    child_assign[i] = random.randrange(truck_count)
            routes, max_dist, total_dist = decode(child_assign)
            new_pop.append((child_assign, routes, max_dist, total_dist))
        new_pop.sort(key=lambda x: (x[2], x[3]))
        population = new_pop
        if population[0][2] < best[2] or (population[0][2] == best[2] and population[0][3] < best[3]):
            best = population[0]
            report_best_vrp(best[1])
    
    return best[1]