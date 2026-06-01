import numpy as np
import random
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    m = n - 1
    k = truck_count
    if k >= m:
        routes = [[0, i, 0] for i in customers]
        while len(routes) < k:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes

    def route_dist(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i]][route[i+1]]
        return d

    def max_dist(routes):
        return max(route_dist(r) for r in routes)

    def build_route(cluster):
        if not cluster:
            return [0, 0]
        route = [0]
        unvisited = set(cluster)
        while unvisited:
            last = route[-1]
            # closest unvisited with tie-breaking
            best = min(unvisited, key=lambda c: (distance_matrix[last][c], c))
            route.append(best)
            unvisited.remove(best)
        route.append(0)
        return route

    # Initialization: random clustering
    pop_size = 20
    population = []
    for _ in range(pop_size):
        # assign each customer to random cluster
        clusters = [[] for _ in range(k)]
        for c in customers:
            cl = random.randrange(k)
            clusters[cl].append(c)
        # ensure no cluster becomes empty? allow empty
        routes = [build_route(clusters[i]) for i in range(k)]
        population.append(routes)

    # Evaluate initial best
    best_routes = min(population, key=lambda rs: (max_dist(rs), sum(route_dist(r) for r in rs), random.random()))
    best_max = max_dist(best_routes)
    report_best_vrp(best_routes)

    # GA parameters
    pop = [list(r) for r in population]  # deep copy
    pop_max = [max_dist(r) for r in pop]
    pop_total = [sum(route_dist(r) for r in rs) for rs in pop]

    gen_max = 100
    for gen in range(gen_max):
        # Tournament selection
        def tournament():
            idx1 = random.randint(0, pop_size-1)
            idx2 = random.randint(0, pop_size-1)
            if pop_max[idx1] < pop_max[idx2] or (pop_max[idx1] == pop_max[idx2] and pop_total[idx1] < pop_total[idx2]):
                return idx1
            else:
                return idx2

        parent1_idx = tournament()
        parent2_idx = tournament()
        parent1 = pop[parent1_idx]
        parent2 = pop[parent2_idx]

        # Assignment crossover
        # Decide route assignment for each customer
        clusters = [[] for _ in range(k)]
        for c in customers:
            # find which route index c belongs to in parent1 and parent2
            idx1 = None
            for ri, route in enumerate(parent1):
                if c in route:
                    idx1 = ri
                    break
            idx2 = None
            for ri, route in enumerate(parent2):
                if c in route:
                    idx2 = ri
                    break
            # if not found (should not happen), assign to random
            if idx1 is None:
                idx1 = random.randrange(k)
            if idx2 is None:
                idx2 = random.randrange(k)
            # choose with equal probability
            chosen = idx1 if random.random() < 0.5 else idx2
            clusters[chosen].append(c)
        # Build routes
        child = [build_route(clusters[i]) for i in range(k)]

        # Mutation (with probability 0.2)
        if random.random() < 0.2:
            # pick random customer and reassign to different cluster
            c = random.choice(customers)
            old_cl = None
            for ri, route in enumerate(child):
                if c in route:
                    old_cl = ri
                    break
            if old_cl is None:
                old_cl = random.randrange(k)
            new_cl = random.randrange(k)
            while new_cl == old_cl and k > 1:
                new_cl = random.randrange(k)
            # remove from old cluster
            clusters[old_cl].remove(c)
            clusters[new_cl].append(c)
            # rebuild affected routes
            child[old_cl] = build_route(clusters[old_cl])
            child[new_cl] = build_route(clusters[new_cl])

        # Evaluate child
        child_max = max_dist(child)
        child_total = sum(route_dist(r) for r in child)

        # Update best
        if child_max < best_max or (child_max == best_max and child_total < sum(route_dist(r) for r in best_routes)):
            best_max = child_max
            best_routes = [list(r) for r in child]
            report_best_vrp(best_routes)

        # Replacement: replace worst individual (or if child better than worst)
        worst_idx = max(range(pop_size), key=lambda i: (pop_max[i], pop_total[i], random.random()))
        if child_max < pop_max[worst_idx] or (child_max == pop_max[worst_idx] and child_total < pop_total[worst_idx]):
            pop[worst_idx] = [list(r) for r in child]
            pop_max[worst_idx] = child_max
            pop_total[worst_idx] = child_total
        else:
            # maybe replace a random individual with poor fitness? Not needed
            pass

    return best_routes