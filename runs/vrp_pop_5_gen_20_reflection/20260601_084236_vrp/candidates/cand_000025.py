import numpy as np
import random
from copy import deepcopy

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n:
        routes = []
        for i in range(1, n):
            routes.append([0, i, 0])
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_distance(route):
        if len(route) < 2:
            return 0.0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def max_distance(routes):
        return max(route_distance(r) for r in routes)

    def two_opt(route):
        if len(route) <= 3:
            return route
        improved = True
        best = route[:]
        best_dist = route_distance(best)
        while improved:
            improved = False
            for i in range(1, len(best)-2):
                for j in range(i+1, len(best)-1):
                    if j-i == 1:
                        continue
                    new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    d = route_distance(new_route)
                    if d < best_dist - 1e-12:
                        best = new_route
                        best_dist = d
                        improved = True
                        break
                if improved:
                    break
        return best

    def construct_initial(seed_idx):
        random.seed(seed_idx)
        seeds = random.sample(customers, min(truck_count, len(customers)))
        clusters = [[] for _ in range(truck_count)]
        for i, s in enumerate(seeds):
            clusters[i].append(s)
        remaining = [c for c in customers if c not in seeds]
        for cust in remaining:
            best_dist = float('inf')
            best_cluster = 0
            for i, seed in enumerate(seeds):
                d = distance_matrix[cust, seed]
                if d < best_dist - 1e-12:
                    best_dist = d
                    best_cluster = i
            clusters[best_cluster].append(cust)
        routes = []
        for cluster in clusters:
            if not cluster:
                routes.append([0, 0])
            else:
                unvisited = set(cluster)
                current = 0
                tour = [0]
                while unvisited:
                    next_cust = min(unvisited, key=lambda c: distance_matrix[current, c])
                    tour.append(next_cust)
                    unvisited.remove(next_cust)
                    current = next_cust
                tour.append(0)
                route = two_opt(tour)
                routes.append(route)
        return routes

    def local_search(routes):
        improved = True
        while improved:
            improved = False
            # relocate
            for truck_a in range(truck_count):
                route_a = routes[truck_a]
                if len(route_a) <= 2:
                    continue
                custs_a = route_a[1:-1]
                for cust in custs_a:
                    new_route_a = [0] + [c for c in route_a[1:-1] if c != cust] + [0]
                    for truck_b in range(truck_count):
                        if truck_b == truck_a:
                            continue
                        route_b = routes[truck_b]
                        best_pos = None
                        best_d = float('inf')
                        for i in range(1, len(route_b)):
                            new_r = route_b[:i] + [cust] + route_b[i:]
                            d = route_distance(new_r)
                            if d < best_d - 1e-12:
                                best_d = d
                                best_pos = i
                        if best_pos is not None:
                            new_route_b = route_b[:best_pos] + [cust] + route_b[best_pos:]
                            new_route_a_opt = two_opt(new_route_a)
                            new_route_b_opt = two_opt(new_route_b)
                            new_routes = routes[:]
                            new_routes[truck_a] = new_route_a_opt
                            new_routes[truck_b] = new_route_b_opt
                            new_max = max_distance(new_routes)
                            old_max = max_distance(routes)
                            if new_max < old_max - 1e-12:
                                routes = new_routes
                                improved = True
                                if new_max < max_distance(routes):
                                    report_best_vrp(routes)
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # swap
            for truck_a in range(truck_count):
                route_a = routes[truck_a]
                if len(route_a) <= 2:
                    continue
                custs_a = route_a[1:-1]
                for truck_b in range(truck_a+1, truck_count):
                    route_b = routes[truck_b]
                    if len(route_b) <= 2:
                        continue
                    custs_b = route_b[1:-1]
                    for cust_a in custs_a:
                        for cust_b in custs_b:
                            new_route_a = [0] + [c for c in route_a[1:-1] if c != cust_a] + [cust_b] + [0]
                            new_route_b = [0] + [c for c in route_b[1:-1] if c != cust_b] + [cust_a] + [0]
                            new_route_a_opt = two_opt(new_route_a)
                            new_route_b_opt = two_opt(new_route_b)
                            new_routes = routes[:]
                            new_routes[truck_a] = new_route_a_opt
                            new_routes[truck_b] = new_route_b_opt
                            new_max = max_distance(new_routes)
                            old_max = max_distance(routes)
                            if new_max < old_max - 1e-12:
                                routes = new_routes
                                improved = True
                                if new_max < max_distance(routes):
                                    report_best_vrp(routes)
                                break
                    if improved:
                        break
                if improved:
                    break
        return routes

    def perturb(routes):
        # focus on longest route
        longest_idx = max(range(truck_count), key=lambda i: route_distance(routes[i]) if len(routes[i])>2 else 0)
        if len(routes[longest_idx]) <= 3:
            return routes
        route = routes[longest_idx]
        # remove a random segment of length 1 to len(route)-2
        seg_len = random.randint(1, len(route)-2)
        start = random.randint(1, len(route)-seg_len-1)
        removed = route[start:start+seg_len]
        new_route = route[:start] + route[start+seg_len:]
        new_routes = routes[:]
        new_routes[longest_idx] = new_route
        # reinsert each removed customer greedily into best route and position to minimize max distance
        for cust in removed:
            best_route_idx = None
            best_pos = None
            best_max = float('inf')
            temp_routes = [r[:] for r in new_routes]
            for t in range(truck_count):
                r = temp_routes[t]
                for i in range(1, len(r)):
                    test_r = r[:i] + [cust] + r[i:]
                    temp_routes[t] = test_r
                    cand_max = max(route_distance(temp_routes[i]) for i in range(truck_count))
                    if cand_max < best_max - 1e-12:
                        best_max = cand_max
                        best_route_idx = t
                        best_pos = i
                    temp_routes[t] = r
            if best_route_idx is not None:
                new_routes[best_route_idx] = new_routes[best_route_idx][:best_pos] + [cust] + new_routes[best_route_idx][best_pos:]
        # re-optimize each modified route with 2-opt
        for idx in range(truck_count):
            new_routes[idx] = two_opt(new_routes[idx])
        return new_routes

    best_routes = None
    best_max = float('inf')
    max_outer = min(3, max(truck_count, 5))  # limit iterations to avoid timeout
    for restart in range(max_outer):
        routes = construct_initial(restart)
        current_max = max_distance(routes)
        if current_max < best_max - 1e-12:
            best_routes = [r[:] for r in routes]
            best_max = current_max
            report_best_vrp(best_routes)
        # local search + perturbation cycle
        for _ in range(3):  # small number of perturbations
            routes = local_search(routes)
            current_max = max_distance(routes)
            if current_max < best_max - 1e-12:
                best_routes = [r[:] for r in routes]
                best_max = current_max
                report_best_vrp(best_routes)
            routes = perturb(routes)
            current_max = max_distance(routes)
            if current_max < best_max - 1e-12:
                best_routes = [r[:] for r in routes]
                best_max = current_max
                report_best_vrp(best_routes)
    return best_routes