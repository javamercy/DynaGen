import random
import math
import numpy as np

def solve_vrp(distance_matrix, truck_count):
    random.seed(0)
    n = distance_matrix.shape[0]
    if truck_count >= n - 1:
        routes = [[0, i, 0] for i in range(1, n)]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_dist(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_route_dist(routes):
        return max(route_dist(r) for r in routes)

    def greedy_insertion(routes, customer):
        best_inc = float('inf')
        best_ri = -1
        best_pos = -1
        for ri, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_dist = route_dist(route[:pos] + [customer] + route[pos:])
                old_dist = route_dist(route)
                inc = new_dist - old_dist
                if inc < best_inc or (inc == best_inc and (ri < best_ri or (ri == best_ri and pos < best_pos))):
                    best_inc = inc
                    best_ri = ri
                    best_pos = pos
        routes[best_ri].insert(best_pos, customer)
        return routes

    def random_construction():
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        for c in customers:
            routes = greedy_insertion(routes, c)
        return routes

    def tournament_select(pop):
        i = random.randrange(len(pop))
        j = random.randrange(len(pop))
        if max_route_dist(pop[i]) < max_route_dist(pop[j]):
            return pop[i]
        else:
            return pop[j]

    def crossover(p1, p2):
        child = [list(r) for r in p1]
        route_indices = list(range(truck_count))
        random.shuffle(route_indices)
        num_take = max(1, truck_count // 2)
        selected = route_indices[:num_take]
        customers_take = set()
        for idx in selected:
            for c in p2[idx]:
                if c != 0:
                    customers_take.add(c)
        for route in child:
            for c in list(route):
                if c != 0 and c in customers_take:
                    route.remove(c)
        all_customers = set(range(1, n))
        child_customers = set()
        for route in child:
            for c in route:
                if c != 0:
                    child_customers.add(c)
        missing = all_customers - child_customers
        for c in missing:
            child = greedy_insertion(child, c)
        while len(child) < truck_count:
            child.append([0, 0])
        return child

    def mutate(routes):
        new_routes = [list(r) for r in routes]
        customers = []
        for ri, route in enumerate(new_routes):
            for pos, c in enumerate(route):
                if c != 0:
                    customers.append((ri, pos, c))
        if not customers:
            return new_routes
        idx = random.randrange(len(customers))
        ri, pos, cust = customers[idx]
        new_routes[ri].pop(pos)
        new_routes = greedy_insertion(new_routes, cust)
        return new_routes

    def local_search(routes):
        best = [list(r) for r in routes]
        best_max = max_route_dist(best)
        improved = True
        iteration = 0
        while improved and iteration < 100:
            improved = False
            iteration += 1
            # intra-route 2-opt
            for ri, route in enumerate(routes):
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        if new_route[0] != 0 or new_route[-1] != 0:
                            continue
                        new_dist = route_dist(new_route)
                        old_dist = route_dist(route)
                        if new_dist < old_dist:
                            routes[ri] = new_route
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                cur_max = max_route_dist(routes)
                if cur_max < best_max:
                    best_max = cur_max
                    best = [list(r) for r in routes]
                    report_best_vrp(best)
                continue
            # inter-route relocate
            for cust in range(1, n):
                ri = None
                pos = None
                for ridx, route in enumerate(routes):
                    if cust in route:
                        ri = ridx
                        pos = route.index(cust)
                        break
                if ri is None:
                    continue
                old_route = routes[ri][:]
                routes[ri].pop(pos)
                best_ri = -1
                best_pos = -1
                best_new_max = float('inf')
                for other_ri, other_route in enumerate(routes):
                    if other_ri == ri:
                        continue
                    for p in range(1, len(other_route)):
                        new_other = other_route[:p] + [cust] + other_route[p:]
                        new_max = max(route_dist(routes[ri]) if len(routes[ri]) > 1 else 0,
                                      route_dist(new_other),
                                      *[route_dist(r) for idx, r in enumerate(routes) if idx not in (ri, other_ri)])
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_ri = other_ri
                            best_pos = p
                if best_new_max < best_max and best_ri != -1:
                    routes[best_ri].insert(best_pos, cust)
                    best_max = best_new_max
                    best = [list(r) for r in routes]
                    improved = True
                    report_best_vrp(best)
                else:
                    routes[ri] = old_route[:]
            if improved:
                continue
            break
        return best

    pop_size = 10
    generations = 10
    population = [random_construction() for _ in range(pop_size)]
    best = min(population, key=max_route_dist)
    for gen in range(generations):
        new_pop = []
        for _ in range(pop_size):
            p1 = tournament_select(population)
            p2 = tournament_select(population)
            offspring = crossover(p1, p2)
            offspring = mutate(offspring)
            offspring = local_search(offspring)
            new_pop.append(offspring)
            cur = max_route_dist(offspring)
            if cur < max_route_dist(best):
                best = [list(r) for r in offspring]
                report_best_vrp(best)
        population = new_pop
        # elitism
        worst = max(population, key=max_route_dist)
        best_val = max_route_dist(best)
        if best_val < max_route_dist(worst):
            population.remove(worst)
            population.append([list(r) for r in best])
    best = local_search(best)
    return best