import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    random.seed(0)
    dist = distance_matrix

    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += dist[route[i], route[i+1]]
        return d

    def insertion_delta(route, pos, cust):
        prev = route[pos-1]
        nxt = route[pos]
        return dist[prev, cust] + dist[cust, nxt] - dist[prev, nxt]

    # Farthest-first initial construction (same as parents but okay)
    customers = sorted(range(1, n), key=lambda c: -dist[0][c])
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    for cust in customers:
        best_truck = None
        best_pos = None
        best_max = float('inf')
        best_total = float('inf')
        for t, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_dist = route_dists[t] + insertion_delta(route, pos, cust)
                new_max = max(route_dists[:t] + [new_dist] + route_dists[t+1:])
                new_total = sum(route_dists) + insertion_delta(route, pos, cust)
                if new_max < best_max or (new_max == best_max and new_total < best_total):
                    best_max = new_max
                    best_total = new_total
                    best_truck = t
                    best_pos = pos
        route = routes[best_truck]
        routes[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
        route_dists[best_truck] += insertion_delta(route, best_pos, cust)

    current_routes = [list(r) for r in routes]
    current_dists = list(route_dists)
    current_max = max(current_dists)
    current_total = sum(current_dists)
    best_routes = [list(r) for r in routes]
    best_dists = list(route_dists)
    best_max = current_max
    best_total = current_total
    report_best_vrp(best_routes)

    # VND: variable neighborhood descent
    def vnd(routes, dists):
        improved = True
        while improved:
            improved = False
            # inter-route swap (exchange)
            for t1 in range(len(routes)):
                for t2 in range(t1+1, len(routes)):
                    r1 = routes[t1]
                    r2 = routes[t2]
                    if len(r1) <= 2 or len(r2) <= 2:
                        continue
                    best_delta = 0
                    best_pair = None
                    for i in range(1, len(r1)-1):
                        for j in range(1, len(r2)-1):
                            c1 = r1[i]
                            c2 = r2[j]
                            # compute new distances after swap
                            new_d1 = dists[t1] - removal_delta(r1, i) + insertion_delta_at(r1, i, c2)
                            new_d2 = dists[t2] - removal_delta(r2, j) + insertion_delta_at(r2, j, c1)
                            new_max = max(dists[:t1] + [new_d1] + dists[t1+1:t2] + [new_d2] + dists[t2+1:])
                            new_total = sum(dists) - removal_delta(r1, i) - removal_delta(r2, j) + insertion_delta_at(r1, i, c2) + insertion_delta_at(r2, j, c1)
                            if new_max < best_max or (new_max == best_max and new_total < best_total):
                                best_max = new_max
                                best_total = new_total
                                best_pair = (t1, i, t2, j)
                    if best_pair is not None:
                        t1, i, t2, j = best_pair
                        routes[t1][i], routes[t2][j] = routes[t2][j], routes[t1][i]
                        dists[t1] = route_distance(routes[t1])
                        dists[t2] = route_distance(routes[t2])
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # inter-route relocate (move)
            for t1 in range(len(routes)):
                for t2 in range(len(routes)):
                    if t1 == t2:
                        continue
                    r1 = routes[t1]
                    r2 = routes[t2]
                    if len(r1) <= 2:
                        continue
                    best_delta = 0
                    best_move = None
                    for i in range(1, len(r1)-1):
                        c = r1[i]
                        new_d1 = dists[t1] - removal_delta(r1, i)
                        for j in range(1, len(r2)-1):
                            new_d2 = dists[t2] + insertion_delta_at(r2, j, c)
                            new_max = max(dists[:t1] + [new_d1] + dists[t1+1:t2] + [new_d2] + dists[t2+1:])
                            new_total = sum(dists) - removal_delta(r1, i) + insertion_delta_at(r2, j, c)
                            if new_max < best_max or (new_max == best_max and new_total < best_total):
                                best_max = new_max
                                best_total = new_total
                                best_move = (t1, i, t2, j)
                    if best_move is not None:
                        t1, i, t2, j = best_move
                        routes[t1].pop(i)
                        if i < j:  # adjust index after removal
                            routes[t2].insert(j, c)
                        else:
                            routes[t2].insert(j, c)
                        dists[t1] = route_distance(routes[t1])
                        dists[t2] = route_distance(routes[t2])
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # intra-route 2-opt
            for t in range(len(routes)):
                route = routes[t]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        if new_dist < dists[t] - 1e-9:
                            new_max = max(dists[:t] + [new_dist] + dists[t+1:])
                            new_total = sum(dists) - dists[t] + new_dist
                            if new_max < best_max or (new_max == best_max and new_total < best_total):
                                dists[t] = new_dist
                                routes[t] = new_route
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # intra-route Or-opt (relocate one customer)
            for t in range(len(routes)):
                route = routes[t]
                if len(route) <= 2:
                    continue
                for i in range(1, len(route)-1):
                    c = route[i]
                    for j in range(1, len(route)-1):
                        if j == i or j == i-1:
                            continue
                        new_route = route[:i] + route[i+1:]
                        new_route = new_route[:j] + [c] + new_route[j:]
                        new_dist = route_distance(new_route)
                        if new_dist < dists[t] - 1e-9:
                            new_max = max(dists[:t] + [new_dist] + dists[t+1:])
                            new_total = sum(dists) - dists[t] + new_dist
                            if new_max < best_max or (new_max == best_max and new_total < best_total):
                                dists[t] = new_dist
                                routes[t] = new_route
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return routes, dists

    # Shaking: random perturbation
    def shake(routes, dists, k):
        for _ in range(k):
            op = random.choice(['swap', 'relocate'])
            if op == 'swap':
                # choose two different trucks
                t1 = random.randrange(len(routes))
                t2 = random.randrange(len(routes))
                while t2 == t1:
                    t2 = random.randrange(len(routes))
                r1 = routes[t1]
                r2 = routes[t2]
                if len(r1) <= 2 or len(r2) <= 2:
                    continue
                i = random.randrange(1, len(r1)-1)
                j = random.randrange(1, len(r2)-1)
                routes[t1][i], routes[t2][j] = routes[t2][j], routes[t1][i]
            else:  # relocate
                t1 = random.randrange(len(routes))
                t2 = random.randrange(len(routes))
                while t2 == t1:
                    t2 = random.randrange(len(routes))
                r1 = routes[t1]
                r2 = routes[t2]
                if len(r1) <= 2:
                    continue
                i = random.randrange(1, len(r1)-1)
                c = routes[t1].pop(i)
                if len(r2) <= 2:
                    routes[t2].insert(1, c)
                else:
                    j = random.randrange(1, len(r2)-1)
                    routes[t2].insert(j, c)
        # recalc dists
        for t in range(len(routes)):
            dists[t] = route_distance(routes[t])
        return routes, dists

    # Main VNS loop
    max_iter = 5 * n
    max_shake = max(1, n // 10)
    for it in range(max_iter):
        # VND on current
        new_routes, new_dists = vnd([list(r) for r in current_routes], list(current_dists))
        new_max = max(new_dists)
        new_total = sum(new_dists)
        if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
            best_max = new_max
            best_total = new_total
            best_routes = [list(r) for r in new_routes]
            best_dists = list(new_dists)
            report_best_vrp(best_routes)
        current_routes = new_routes
        current_dists = new_dists
        # Shake
        k = (it % max_shake) + 1
        current_routes, current_dists = shake([list(r) for r in current_routes], list(current_dists), k)

    return best_routes

# Helper functions used within the code
def removal_delta(route, pos):
    prev = route[pos-1]
    nxt = route[pos+1]
    return dist[prev, route[pos]] + dist[route[pos], nxt] - dist[prev, nxt]

def insertion_delta_at(route, pos, cust):
    prev = route[pos-1]
    nxt = route[pos]
    return dist[prev, cust] + dist[cust, nxt] - dist[prev, nxt]