import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    random.seed(0)

    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def total_dist(routes):
        return sum(route_dist(r) for r in routes)

    def best_insertion(c, routes, route_dists):
        best = (float('inf'), -1, -1)
        second = (float('inf'), -1, -1)
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

    def construct_solution(use_random=False):
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        unassigned = list(range(1, n))
        while unassigned:
            bests = []
            for c in unassigned:
                best_new_max, best_route, best_pos, second_new_max = best_insertion(c, routes, route_dists)
                if best_route == -1:
                    continue
                regret = second_new_max - best_new_max if second_new_max != float('inf') else float('inf')
                if use_random:
                    bests.append((random.random(), c, best_route, best_pos, best_new_max))
                else:
                    bests.append((-regret, c, best_route, best_pos, best_new_max))
            bests.sort(key=lambda x: (x[0], x[1]))
            _, c, best_route, best_pos, new_max = bests[0]
            route = routes[best_route]
            route.insert(best_pos, c)
            route_dists[best_route] = route_dist(route)
            unassigned.remove(c)
        return routes, route_dists

    def improve(routes, route_dists):
        improved = True
        while improved:
            improved = False
            # Intra-route 2-opt
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                        new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                        if new < old - 1e-12:
                            route[i:j+1] = reversed(route[i:j+1])
                            route_dists[r_idx] = route_dist(route)
                            improved = True
                            break
                    if improved:
                        break
            # Best-improvement relocate with tie-breaking on total
            max_dist = max(route_dists)
            max_idx = route_dists.index(max_dist)
            best_move = None
            best_new_max = max_dist
            best_new_total = total_dist(routes)
            route_max = routes[max_idx]
            for i in range(1, len(route_max)-1):
                c = route_max[i]
                pred = route_max[i-1]
                succ = route_max[i+1]
                new_max_dist = route_dists[max_idx] - distance_matrix[pred, c] - distance_matrix[c, succ] + distance_matrix[pred, succ]
                for other_idx in range(truck_count):
                    if other_idx == max_idx:
                        continue
                    other_route = routes[other_idx]
                    for pos in range(1, len(other_route)):
                        pred_o = other_route[pos-1]
                        succ_o = other_route[pos]
                        new_other = route_dists[other_idx] - distance_matrix[pred_o, succ_o] + distance_matrix[pred_o, c] + distance_matrix[c, succ_o]
                        other_max = 0.0
                        for j, d in enumerate(route_dists):
                            if j != max_idx and j != other_idx and d > other_max:
                                other_max = d
                        new_overall = max(other_max, new_max_dist, new_other)
                        if new_overall < best_new_max - 1e-12:
                            best_new_max = new_overall
                            best_new_total = total_dist(routes) - route_dists[max_idx] - route_dists[other_idx] + new_max_dist + new_other
                            best_move = (i, other_idx, pos, new_max_dist, new_other)
                        elif abs(new_overall - best_new_max) < 1e-12:
                            new_total = total_dist(routes) - route_dists[max_idx] - route_dists[other_idx] + new_max_dist + new_other
                            if new_total < best_new_total - 1e-12:
                                best_new_max = new_overall
                                best_new_total = new_total
                                best_move = (i, other_idx, pos, new_max_dist, new_other)
            if best_move is not None:
                i, other_idx, pos, new_max_dist, new_other = best_move
                c = route_max.pop(i)
                routes[other_idx].insert(pos, c)
                route_dists[max_idx] = new_max_dist
                route_dists[other_idx] = new_other
                improved = True
                continue
            # Best swap with tie-breaking on total
            max_dist = max(route_dists)
            max_idx = route_dists.index(max_dist)
            best_swap = None
            best_new_max = max_dist
            best_new_total = total_dist(routes)
            route_max = routes[max_idx]
            for i in range(1, len(route_max)-1):
                c1 = route_max[i]
                for other_idx in range(truck_count):
                    if other_idx == max_idx:
                        continue
                    other_route = routes[other_idx]
                    for j in range(1, len(other_route)-1):
                        c2 = other_route[j]
                        old1 = route_dists[max_idx]
                        old2 = route_dists[other_idx]
                        pred1 = route_max[i-1]
                        succ1 = route_max[i+1]
                        new_dist_max = old1 - distance_matrix[pred1, c1] - distance_matrix[c1, succ1] + distance_matrix[pred1, c2] + distance_matrix[c2, succ1]
                        pred2 = other_route[j-1]
                        succ2 = other_route[j+1]
                        new_dist_other = old2 - distance_matrix[pred2, c2] - distance_matrix[c2, succ2] + distance_matrix[pred2, c1] + distance_matrix[c1, succ2]
                        other_max = 0.0
                        for k, d in enumerate(route_dists):
                            if k != max_idx and k != other_idx and d > other_max:
                                other_max = d
                        new_overall = max(other_max, new_dist_max, new_dist_other)
                        if new_overall < best_new_max - 1e-12:
                            best_new_max = new_overall
                            best_new_total = total_dist(routes) - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                            best_swap = (i, other_idx, j, new_dist_max, new_dist_other)
                        elif abs(new_overall - best_new_max) < 1e-12:
                            new_total = total_dist(routes) - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                            if new_total < best_new_total - 1e-12:
                                best_new_max = new_overall
                                best_new_total = new_total
                                best_swap = (i, other_idx, j, new_dist_max, new_dist_other)
            if best_swap is not None:
                i, other_idx, j, new_dist_max, new_dist_other = best_swap
                c1 = route_max[i]
                c2 = routes[other_idx][j]
                route_max[i] = c2
                routes[other_idx][j] = c1
                route_dists[max_idx] = new_dist_max
                route_dists[other_idx] = new_dist_other
                improved = True
                continue
            # Best 2-opt* with tie-breaking on total
            max_dist = max(route_dists)
            max_idx = route_dists.index(max_dist)
            best_cross = None
            best_new_max = max_dist
            best_new_total = total_dist(routes)
            route_max = routes[max_idx]
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                for i in range(1, len(route_max)-1):
                    for j in range(1, len(other_route)-1):
                        old1 = distance_matrix[route_max[i], route_max[i+1]]
                        old2 = distance_matrix[other_route[j], other_route[j+1]]
                        new1 = distance_matrix[route_max[i], other_route[j+1]]
                        new2 = distance_matrix[other_route[j], route_max[i+1]]
                        new_dist_max = route_dists[max_idx] - old1 + new1
                        new_dist_other = route_dists[other_idx] - old2 + new2
                        other_max = 0.0
                        for k, d in enumerate(route_dists):
                            if k != max_idx and k != other_idx and d > other_max:
                                other_max = d
                        new_overall = max(other_max, new_dist_max, new_dist_other)
                        if new_overall < best_new_max - 1e-12:
                            best_new_max = new_overall
                            best_new_total = total_dist(routes) - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                            best_cross = (i, other_idx, j, new_dist_max, new_dist_other)
                        elif abs(new_overall - best_new_max) < 1e-12:
                            new_total = total_dist(routes) - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                            if new_total < best_new_total - 1e-12:
                                best_new_max = new_overall
                                best_new_total = new_total
                                best_cross = (i, other_idx, j, new_dist_max, new_dist_other)
            if best_cross is not None:
                i, other_idx, j, new_dist_max, new_dist_other = best_cross
                route_max_new = route_max[:i+1] + routes[other_idx][j+1:]
                other_route_new = routes[other_idx][:j+1] + route_max[i+1:]
                routes[max_idx] = route_max_new
                routes[other_idx] = other_route_new
                route_dists[max_idx] = route_dist(route_max_new)
                route_dists[other_idx] = route_dist(other_route_new)
                improved = True
                continue
            break  # no improvement found
        return routes, route_dists

    # Population initialization
    pop_size = min(20, max(10, n//5))
    population = []
    for _ in range(pop_size - 1):
        routes, route_dists = construct_solution(use_random=True)
        population.append((routes, max(route_dists), total_dist(routes)))
    # Add one deterministic
    routes, route_dists = construct_solution(use_random=False)
    population.append((routes, max(route_dists), total_dist(routes)))
    # Sort by max, then total
    population.sort(key=lambda x: (x[1], x[2]))
    best_routes = population[0][0]
    best_max = population[0][1]
    best_total = population[0][2]
    report_best_vrp(best_routes)

    # Main GA loop
    max_gen = min(50, n*2)
    pc = 0.8
    for gen in range(max_gen):
        new_pop = []
        for _ in range(pop_size):
            # Tournament selection
            tournament = random.sample(population, min(3, len(population)))
            parent1 = min(tournament, key=lambda x: (x[1], x[2]))[0]
            tournament = random.sample(population, min(3, len(population)))
            parent2 = min(tournament, key=lambda x: (x[1], x[2]))[0]
            # Crossover
            if random.random() < pc:
                # Route crossover: copy a subset of routes from parent1, then repair
                child_routes = [None] * truck_count
                used = set()
                # Randomly select a set of routes from parent1
                routes1 = parent1
                selected_indices = random.sample(range(truck_count), random.randint(1, truck_count//2))
                for idx in selected_indices:
                    route = routes1[idx][:]
                    child_routes[idx] = route
                    for c in route[1:-1]:
                        used.add(c)
                # Fill remaining routes with routes from parent2 that don't conflict, then repair
                routes2 = parent2
                remaining_indices = [i for i in range(truck_count) if child_routes[i] is None]
                for idx in remaining_indices:
                    route = routes2[idx][:]
                    # Remove already used customers
                    new_route = [0]
                    for c in route[1:-1]:
                        if c not in used:
                            new_route.append(c)
                            used.add(c)
                    new_route.append(0)
                    child_routes[idx] = new_route
                # Now insert missing customers using best insertion
                unassigned = [c for c in range(1, n) if c not in used]
                child_route_dists = [route_dist(r) for r in child_routes]
                while unassigned:
                    bests = []
                    for c in unassigned:
                        best_new_max, best_route, best_pos, _ = best_insertion(c, child_routes, child_route_dists)
                        if best_route == -1:
                            continue
                        bests.append((best_new_max, c, best_route, best_pos))
                    if not bests:
                        break
                    bests.sort(key=lambda x: (x[0], x[1]))
                    _, c, best_route, best_pos = bests[0]
                    child_routes[best_route].insert(best_pos, c)
                    child_route_dists[best_route] = route_dist(child_routes[best_route])
                    unassigned.remove(c)
            else:
                child_routes = [route[:] for route in parent1]
            # Local search mutation
            child_routes, child_route_dists = improve(child_routes, [route_dist(r) for r in child_routes])
            child_max = max(child_route_dists)
            child_total = total_dist(child_routes)
            new_pop.append((child_routes, child_max, child_total))
            # Update best
            if child_max < best_max - 1e-12 or (abs(child_max - best_max) < 1e-12 and child_total < best_total - 1e-12):
                best_routes = [route[:] for route in child_routes]
                best_max = child_max
                best_total = child_total
                report_best_vrp(best_routes)
        # Merge new and old, keep best pop_size
        combined = population + new_pop
        combined.sort(key=lambda x: (x[1], x[2]))
        population = combined[:pop_size]
    return best_routes