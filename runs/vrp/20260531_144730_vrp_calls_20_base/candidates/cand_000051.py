import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    customers = list(range(1, n))
    INF = 1e15

    def route_dist(route):
        d = 0.0
        for i in range(len(route) - 1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def two_opt(route):
        best = route[:]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best) - 2):
                for j in range(i + 1, len(best) - 1):
                    if j - i == 1:
                        continue
                    new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    if route_dist(new_route) < route_dist(best) - 1e-12:
                        best = new_route
                        improved = True
                        break
                if improved:
                    break
        return best

    def best_insertion(c, routes, route_dists):
        best = (INF, -1, -1)
        second = (INF, -1, -1)
        for r_idx, route in enumerate(routes):
            if len(route) < 2:
                continue
            other_max = 0.0
            for j, d in enumerate(route_dists):
                if j != r_idx and d > other_max:
                    other_max = d
            for pos in range(1, len(route)):
                pred = route[pos-1]
                succ = route[pos]
                new_dist = route_dists[r_idx] - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
                new_max = max(other_max, new_dist)
                if new_max < best[0]:
                    best, second = (new_max, r_idx, pos), best
                elif new_max < second[0]:
                    second = (new_max, r_idx, pos)
        return best[0], best[1], best[2], second[0]

    def construct_solution(seed):
        rng = random.Random(seed)
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        unassigned = set(customers)
        while unassigned:
            candidates = []
            for c in unassigned:
                best_val, best_route, best_pos, second_val = best_insertion(c, routes, route_dists)
                if best_route == -1:
                    continue
                regret = second_val - best_val if second_val != INF else INF
                candidates.append((-regret, c, best_route, best_pos, best_val))
            if not candidates:
                break
            # sort by regret descending, then by customer index for determinism
            candidates.sort(key=lambda x: (x[0], x[1]))
            # random top-3 for diversity
            top_k = min(3, len(candidates))
            idx = rng.randint(0, top_k - 1)
            _, c, best_route, best_pos, _ = candidates[idx]
            routes[best_route].insert(best_pos, c)
            route_dists[best_route] = route_dist(routes[best_route])
            unassigned.remove(c)
        # apply 2-opt on each route
        for r_idx in range(truck_count):
            routes[r_idx] = two_opt(routes[r_idx])
            route_dists[r_idx] = route_dist(routes[r_idx])
        return routes, max(route_dists)

    # Population parameters
    pop_size = min(10, max(2, truck_count * 2))
    max_gen = max(20, n * 2)

    # Initialize population
    population = []
    best_routes = None
    best_max = INF
    for i in range(pop_size):
        routes, mx = construct_solution(i)
        population.append((routes, mx))
        if mx < best_max - 1e-12:
            best_max = mx
            best_routes = [route[:] for route in routes]
            report_best_vrp(best_routes)

    # Genetic algorithm
    random.seed(0)  # deterministic global random
    for gen in range(max_gen):
        # Tournament selection
        parents = []
        for _ in range(2):
            idx1, idx2 = random.sample(range(pop_size), 2)
            if population[idx1][1] < population[idx2][1]:
                parents.append(population[idx1])
            else:
                parents.append(population[idx2])
        parent1, parent2 = parents

        # BCRC crossover
        # Choose random subset of customers from parent1 (size up to 30% of customers)
        num_cust = max(1, int(len(customers) * 0.3))
        subset = random.sample(customers, num_cust)
        # Remove subset customers from parent1 routes to get partial
        temp_routes = [route[:] for route in parent1[0]]
        temp_dists = [route_dist(r) for r in temp_routes]
        for c in subset:
            for r in temp_routes:
                if c in r:
                    pos = r.index(c)
                    pred = r[pos-1]
                    succ = r[pos+1]
                    r.pop(pos)
                    # recalc dist (quick but not efficient; we'll recalc later)
                    break
        # recalc distances after removal
        temp_dists = [route_dist(r) for r in temp_routes]
        # For each customer in subset, insert into parent2 routes using regret-2
        offspring_routes = [route[:] for route in parent2[0]]
        offspring_dists = [route_dist(r) for r in offspring_routes]
        for c in subset:
            best_val, best_route, best_pos, _ = best_insertion(c, offspring_routes, offspring_dists)
            if best_route != -1:
                offspring_routes[best_route].insert(best_pos, c)
                offspring_dists[best_route] = route_dist(offspring_routes[best_route])
        # Mutation: ruin-recreate with probability 0.5
        if random.random() < 0.5 and len(customers) > 1:
            ruin_size = random.randint(1, max(1, len(customers) // 5))
            to_ruin = random.sample(customers, ruin_size)
            # remove from offspring
            for c in to_ruin:
                for r in offspring_routes:
                    if c in r:
                        pos = r.index(c)
                        r.pop(pos)
                        break
            offspring_dists = [route_dist(r) for r in offspring_routes]
            # reinsert using regret-2 with random tie-breaking
            unrouted = set(to_ruin)
            while unrouted:
                candidates = []
                for c in unrouted:
                    best_val, best_route, best_pos, second_val = best_insertion(c, offspring_routes, offspring_dists)
                    if best_route == -1:
                        continue
                    regret = second_val - best_val if second_val != INF else INF
                    candidates.append((-regret, c, best_route, best_pos, best_val))
                if not candidates:
                    break
                candidates.sort(key=lambda x: (x[0], x[1]))
                # choose from top-3 randomly
                top_k = min(3, len(candidates))
                idx = random.randint(0, top_k - 1)
                _, c, best_route, best_pos, _ = candidates[idx]
                offspring_routes[best_route].insert(best_pos, c)
                offspring_dists[best_route] = route_dist(offspring_routes[best_route])
                unrouted.remove(c)
        # Apply 2-opt on each route
        for r_idx in range(truck_count):
            if len(offspring_routes[r_idx]) < 4:
                continue
            route_before = offspring_routes[r_idx][:]
            offspring_routes[r_idx] = two_opt(offspring_routes[r_idx])
            if route_dist(offspring_routes[r_idx]) < route_dist(route_before):
                pass
        offspring_dists = [route_dist(r) for r in offspring_routes]
        offspring_max = max(offspring_dists)

        # Replace worst in population (if better)
        worst_idx = max(range(pop_size), key=lambda i: population[i][1])
        if offspring_max < population[worst_idx][1] - 1e-12:
            population[worst_idx] = ([route[:] for route in offspring_routes], offspring_max)
            # Update global best
            if offspring_max < best_max - 1e-12:
                best_max = offspring_max
                best_routes = [route[:] for route in offspring_routes]
                report_best_vrp(best_routes)

    # Ensure all routes start and end at 0, no empty routes except [0,0]
    for r in best_routes:
        if r[0] != 0:
            r.insert(0, 0)
        if r[-1] != 0:
            r.append(0)
    # Verify feasibility
    assigned = []
    for r in best_routes:
        assigned.extend(r[1:-1])
    assert len(set(assigned)) == n - 1
    return best_routes