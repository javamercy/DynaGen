import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))

    def route_length(route):
        if len(route) <= 1:
            return 0.0
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    best_max = float('inf')
    best_routes = None

    def report_best_vrp(routes):
        nonlocal best_max, best_routes
        m = max(route_length(r) for r in routes)
        if m < best_max - 1e-12:
            best_max = m
            best_routes = [list(r) for r in routes]

    def decode(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        lengths = [0.0] * truck_count
        for cust in perm:
            best_max_val = float('inf')
            best_inc = float('inf')
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
                    inc = new_len - lengths[r]
                    if new_max < best_max_val or (abs(new_max - best_max_val) < 1e-12 and inc < best_inc):
                        best_max_val = new_max
                        best_inc = inc
                        best_r = r
                        best_p = p
            routes[best_r].insert(best_p, cust)
            lengths[best_r] = route_length(routes[best_r])
        return routes, lengths, max(lengths)

    def greedy_local_search(routes, lengths):
        max_iter = 2 * n
        for _ in range(max_iter):
            improved = False
            # relocate from longest route
            max_len = max(lengths)
            longest = [i for i, l in enumerate(lengths) if abs(l - max_len) < 1e-12]
            if longest:
                t1 = random.choice(longest)
                route1 = routes[t1]
                if len(route1) > 2:
                    i = random.randint(1, len(route1)-2)
                    cust = route1[i]
                    t2 = random.randint(0, truck_count-1)
                    if t2 != t1:
                        route2 = routes[t2]
                        j = random.randint(1, len(route2)-1)
                        new_route1 = route1[:i] + route1[i+1:]
                        new_len1 = route_length(new_route1)
                        new_route2 = route2[:j] + [cust] + route2[j:]
                        new_len2 = route_length(new_route2)
                        new_max = new_len1
                        for k in range(truck_count):
                            if k == t1:
                                if new_len1 > new_max: new_max = new_len1
                            elif k == t2:
                                if new_len2 > new_max: new_max = new_len2
                            else:
                                if lengths[k] > new_max: new_max = lengths[k]
                        if new_max < max_len - 1e-12:
                            routes[t1] = new_route1
                            lengths[t1] = new_len1
                            routes[t2] = new_route2
                            lengths[t2] = new_len2
                            improved = True
                            report_best_vrp(routes)
            # 2-opt on longest route
            if not improved:
                max_len = max(lengths)
                longest = [i for i, l in enumerate(lengths) if abs(l - max_len) < 1e-12]
                if longest:
                    t = random.choice(longest)
                    route = routes[t]
                    if len(route) > 3:
                        i = random.randint(1, len(route)-3)
                        j = random.randint(i+1, len(route)-2)
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_len = route_length(new_route)
                        new_max = new_len
                        for k in range(truck_count):
                            if k != t and lengths[k] > new_max:
                                new_max = lengths[k]
                        if new_max < max_len - 1e-12:
                            routes[t] = new_route
                            lengths[t] = new_len
                            improved = True
                            report_best_vrp(routes)
            if not improved:
                break
        return routes, lengths

    restarts = max(10, 10 * n)
    for _ in range(restarts):
        perm = customers[:]
        random.shuffle(perm)
        routes, lengths, _ = decode(perm)
        routes, lengths = greedy_local_search(routes, lengths)
        report_best_vrp(routes)

    return best_routes