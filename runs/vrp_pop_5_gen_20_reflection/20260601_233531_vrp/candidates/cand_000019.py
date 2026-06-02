import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]

    customer_count = n - 1
    customers = list(range(1, n))

    # Precompute sorted customers by distance from depot
    depot_distances = [distance_matrix[0, c] for c in customers]
    sorted_customers = [c for _, c in sorted(zip(depot_distances, customers))]

    def decode(assignment):
        # assignment: list of length customer_count, each entry is truck index
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
                # nearest neighbor construction starting from depot
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
        # 2-opt improvement on each route (bounded iterations)
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
        # Inter-route balancing: try to move customers from max route to others
        max_idx = max(range(truck_count), key=lambda i: dists[i])
        min_idx = min(range(truck_count), key=lambda i: dists[i])
        if truck_count >= 2 and max_idx != min_idx:
            # Try moving each customer from max route to min route
            best_routes = [list(r) for r in routes]
            best_max = dists[max_idx]
            for cust in routes[max_idx][1:-1]:  # exclude depots
                # remove cust from max route
                new_max_route = [0] + [c for c in routes[max_idx][1:-1] if c != cust] + [0]
                new_min_route = routes[min_idx][:-1] + [cust, 0]  # insert before depot
                # compute new distances
                new_max_dist = sum(distance_matrix[new_max_route[k], new_max_route[k+1]] for k in range(len(new_max_route)-1))
                new_min_dist = sum(distance_matrix[new_min_route[k], new_min_route[k+1]] for k in range(len(new_min_route)-1))
                new_max_overall = max(new_max_dist, new_min_dist, max(dists[i] for i in range(truck_count) if i not in (max_idx, min_idx)))
                if new_max_overall < best_max:
                    best_max = new_max_overall
                    best_routes[max_idx] = new_max_route
                    best_routes[min_idx] = new_min_route
                    dists[max_idx] = new_max_dist
                    dists[min_idx] = new_min_dist
            routes = best_routes
        max_dist = max(dists) if dists else 0.0
        return routes, max_dist

    # Population parameters
    pop_size = min(50, max(20, customer_count * 2))
    generations = min(100, max(20, customer_count * 2))
    mutation_rate = 1.0 / customer_count if customer_count > 0 else 0.1

    # Initialize population
    population = []
    for _ in range(pop_size):
        assignment = [random.randrange(truck_count) for _ in range(customer_count)]
        routes, max_dist = decode(assignment)
        population.append((assignment, routes, max_dist))

    population.sort(key=lambda x: x[2])
    best = population[0]
    report_best_vrp(best[1])

    for gen in range(generations):
        new_pop = []
        # Elitism: keep best 2
        new_pop.extend(population[:2])
        while len(new_pop) < pop_size:
            # binary tournament selection
            i1, i2 = random.sample(range(pop_size), 2)
            parent1 = population[i1] if population[i1][2] < population[i2][2] else population[i2]
            i3, i4 = random.sample(range(pop_size), 2)
            parent2 = population[i3] if population[i3][2] < population[i4][2] else population[i4]
            # Grouping crossover on sorted customers
            num_segments = max(2, int(math.sqrt(customer_count)))
            segment_length = customer_count // num_segments
            child_assign = [-1] * customer_count
            # Map from customer to index in sorted order
            sorted_index = {c: idx for idx, c in enumerate(sorted_customers)}
            for s in range(num_segments):
                start = s * segment_length
                end = min((s+1) * segment_length, customer_count)
                if s == num_segments - 1:
                    end = customer_count
                parent = parent1 if random.random() < 0.5 else parent2
                for idx in range(start, end):
                    cust = sorted_customers[idx]
                    original_idx = customers.index(cust)
                    child_assign[original_idx] = parent[0][original_idx]
            # Fill any unassigned (should not happen)
            for i in range(customer_count):
                if child_assign[i] == -1:
                    child_assign[i] = random.randrange(truck_count)
            # mutation
            for i in range(customer_count):
                if random.random() < mutation_rate:
                    child_assign[i] = random.randrange(truck_count)
            routes, max_dist = decode(child_assign)
            new_pop.append((child_assign, routes, max_dist))
        new_pop.sort(key=lambda x: x[2])
        population = new_pop
        if population[0][2] < best[2]:
            best = population[0]
            report_best_vrp(best[1])

    return best[1]