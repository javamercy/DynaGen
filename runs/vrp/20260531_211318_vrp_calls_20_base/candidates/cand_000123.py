import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0,0] for _ in range(truck_count)]
    random.seed(0)
    dist = distance_matrix

    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += dist[route[i], route[i+1]]
        return d

    def farthest_insertion(customers):
        route = [0, 0]
        unvisited = set(customers)
        while unvisited:
            best_cust = None
            best_dist = -1
            for cust in unvisited:
                min_dist = min(dist[route[i], cust] for i in range(len(route)))
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_cust = cust
            best_pos = None
            best_inc = float('inf')
            for pos in range(1, len(route)):
                inc = dist[route[pos-1], best_cust] + dist[best_cust, route[pos]] - dist[route[pos-1], route[pos]]
                if inc < best_inc:
                    best_inc = inc
                    best_pos = pos
            route = route[:best_pos] + [best_cust] + route[best_pos:]
            unvisited.remove(best_cust)
        return route

    def build_routes(assignment):
        trucks = [[] for _ in range(truck_count)]
        for idx, cust in enumerate(range(1, n)):
            trucks[assignment[idx]].append(cust)
        routes = []
        dists = []
        for cust_list in trucks:
            if not cust_list:
                routes.append([0,0])
                dists.append(0.0)
            else:
                route = farthest_insertion(cust_list)
                routes.append(route)
                dists.append(route_distance(route))
        return routes, dists

    def evaluate(assignment):
        routes, dists = build_routes(assignment)
        maxd = max(dists)
        total = sum(dists)
        return maxd, total, routes, dists

    def random_assignment():
        return [random.randrange(truck_count) for _ in range(n-1)]

    def crossover(parent1, parent2):
        child = []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child

    def mutate(assignment):
        idx = random.randrange(len(assignment))
        new_truck = random.randrange(truck_count)
        assignment[idx] = new_truck
        return assignment

    def tournament(pop, k=3):
        best = None
        best_fit = float('inf')
        for _ in range(k):
            idx = random.randrange(len(pop))
            fit = pop[idx][0]
            if fit < best_fit:
                best_fit = fit
                best = pop[idx]
        return best[2]

    pop_size = min(100, max(50, 10*n))
    max_gen = min(100, 5*n)
    mut_rate = 0.1

    population = []
    for _ in range(pop_size):
        assignment = random_assignment()
        maxd, total, routes, dists = evaluate(assignment)
        population.append((maxd, total, assignment, routes, dists))
    population.sort(key=lambda x: (x[0], x[1]))

    best_maxd, best_total, best_assignment, best_routes, best_dists = population[0]
    report_best_vrp(best_routes)

    # greedy initial (from farthest-first construction)
    def greedy_initial():
        customers = sorted(range(1, n), key=lambda c: -dist[0][c])
        routes = [[0,0] for _ in range(truck_count)]
        route_dists = [0.0]*truck_count
        for cust in customers:
            best_truck = None
            best_pos = None
            best_new_max = float('inf')
            best_new_total = float('inf')
            for t, route in enumerate(routes):
                for pos in range(1, len(route)):
                    delta = dist[route[pos-1], cust] + dist[cust, route[pos]] - dist[route[pos-1], route[pos]]
                    new_dist = route_dists[t] + delta
                    new_max = max(route_dists[:t] + [new_dist] + route_dists[t+1:])
                    new_total = sum(route_dists) + delta
                    if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                        best_new_max = new_max
                        best_new_total = new_total
                        best_truck = t
                        best_pos = pos
            delta = dist[route[pos-1], cust] + dist[cust, route[pos]] - dist[route[pos-1], route[pos]]
            route = routes[best_truck]
            routes[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
            route_dists[best_truck] += delta
        assignment = [0]*(n-1)
        for t, route in enumerate(routes):
            for cust in route[1:-1]:
                assignment[cust-1] = t
        maxd, total, r, d = evaluate(assignment)
        return assignment, maxd, total, r, d

    greedy_assignment, g_maxd, g_total, g_routes, g_dists = greedy_initial()
    if g_maxd < population[-1][0] or (g_maxd == population[-1][0] and g_total < population[-1][1]):
        population[-1] = (g_maxd, g_total, greedy_assignment, g_routes, g_dists)
    if g_maxd < best_maxd or (g_maxd == best_maxd and g_total < best_total):
        best_maxd, best_total, best_assignment, best_routes, best_dists = g_maxd, g_total, greedy_assignment, g_routes, g_dists
        report_best_vrp(best_routes)

    for gen in range(max_gen):
        new_pop = [population[0], population[1]]
        while len(new_pop) < pop_size:
            parent1 = tournament(population)
            parent2 = tournament(population)
            child_ass = crossover(parent1, parent2)
            if random.random() < mut_rate:
                mutate(child_ass)
            maxd, total, routes, dists = evaluate(child_ass)
            new_pop.append((maxd, total, child_ass, routes, dists))
        population = sorted(new_pop, key=lambda x: (x[0], x[1]))
        if population[0][0] < best_maxd - 1e-9 or (abs(population[0][0] - best_maxd) < 1e-9 and population[0][1] < best_total):
            best_maxd, best_total, best_assignment, best_routes, best_dists = population[0]
            report_best_vrp(best_routes)

    for _ in range(5):
        improved = False
        for t, route in enumerate(best_routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dists[t] - 1e-9:
                        new_max = max(best_dists[:t] + [new_dist] + best_dists[t+1:])
                        new_total = sum(best_dists[:t]) + new_dist + sum(best_dists[t+1:])
                        if new_max < best_maxd - 1e-9 or (abs(new_max - best_maxd) < 1e-9 and new_total < best_total):
                            best_routes[t] = new_route
                            best_dists[t] = new_dist
                            best_maxd = new_max
                            best_total = new_total
                            report_best_vrp(best_routes)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return best_routes