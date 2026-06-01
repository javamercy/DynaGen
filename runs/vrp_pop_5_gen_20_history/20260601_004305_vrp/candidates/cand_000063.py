import numpy as np
import math
import random

def solve_vrp(distance_matrix, truck_count):
    random.seed(0)
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    m = n - 1
    if truck_count >= m:
        routes = [[0, i, 0] for i in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    # --- Construction: greedy insertion based on max route distance ---
    routes = [[0, 0] for _ in range(truck_count)]
    # sort customers by distance from depot descending (ties by index)
    cust_order = sorted(customers, key=lambda c: (-distance_matrix[0][c], c))
    for cust in cust_order:
        best_max = math.inf
        best_route = -1
        best_pos = -1
        for r in range(truck_count):
            route = routes[r]
            for pos in range(1, len(route)):
                # insert cust at position pos
                new_route = route[:pos] + [cust] + route[pos:]
                # compute distances for new route
                new_dist = sum(distance_matrix[new_route[k]][new_route[k+1]] for k in range(len(new_route)-1))
                # compute max distance among all routes
                max_dist = new_dist
                for rr in range(truck_count):
                    if rr == r:
                        continue
                    dist = sum(distance_matrix[routes[rr][k]][routes[rr][k+1]] for k in range(len(routes[rr])-1))
                    if dist > max_dist:
                        max_dist = dist
                if max_dist < best_max or (max_dist == best_max and (r < best_route or (r == best_route and pos < best_pos))):
                    best_max = max_dist
                    best_route = r
                    best_pos = pos
        # apply best insertion
        route = routes[best_route]
        routes[best_route] = route[:best_pos] + [cust] + route[best_pos:]

    # Helper functions (same as parent)
    def route_dist(route):
        return sum(distance_matrix[route[k]][route[k+1]] for k in range(len(route)-1))

    def compute_max():
        return max(route_dist(r) for r in routes)

    def copy_routes():
        return [list(r) for r in routes]

    best_routes = copy_routes()
    best_max = compute_max()
    report_best_vrp(best_routes)

    # Operator definitions (same as parent)
    def op_2opt():
        nonlocal routes, best_max, best_routes
        for ri, route in enumerate(routes):
            if len(route) <= 3:
                continue
            best_local_dist = route_dist(route)
            best_local_route = route[:]
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist(new_route)
                    if new_dist < best_local_dist - 1e-9:
                        best_local_dist = new_dist
                        best_local_route = new_route
                        improved = True
            if improved:
                routes[ri] = best_local_route
                new_max = compute_max()
                if new_max < best_max:
                    best_max = new_max
                    best_routes = copy_routes()
                    report_best_vrp(best_routes)
                return True
        return False

    def op_relocate():
        nonlocal routes, best_max, best_routes
        dists = [route_dist(r) for r in routes]
        longest_idx = max(range(len(routes)), key=lambda i: dists[i])
        src_route = routes[longest_idx]
        if len(src_route) <= 2:
            return False
        for pos_i in range(1, len(src_route) - 1):
            cust = src_route[pos_i]
            for dst_idx in range(len(routes)):
                if dst_idx == longest_idx:
                    continue
                dst_route = routes[dst_idx]
                for pos_j in range(1, len(dst_route)):
                    new_src = src_route[:pos_i] + src_route[pos_i+1:]
                    new_dst = dst_route[:pos_j] + [cust] + dst_route[pos_j:]
                    new_dists = [route_dist(r) for ri, r in enumerate(routes) if ri not in (longest_idx, dst_idx)]
                    new_dist_src = route_dist(new_src)
                    new_dist_dst = route_dist(new_dst)
                    new_max = max([new_dist_src, new_dist_dst] + new_dists)
                    if new_max < compute_max() - 1e-9:
                        routes[longest_idx] = new_src
                        routes[dst_idx] = new_dst
                        if new_max < best_max:
                            best_max = new_max
                            best_routes = copy_routes()
                            report_best_vrp(best_routes)
                        return True
        return False

    def op_swap():
        nonlocal routes, best_max, best_routes
        dists = [route_dist(r) for r in routes]
        longest_idx = max(range(len(routes)), key=lambda i: dists[i])
        src_route = routes[longest_idx]
        if len(src_route) <= 2:
            return False
        for pos_i in range(1, len(src_route) - 1):
            cust_i = src_route[pos_i]
            for dst_idx in range(len(routes)):
                if dst_idx == longest_idx:
                    continue
                dst_route = routes[dst_idx]
                if len(dst_route) <= 2:
                    continue
                for pos_j in range(1, len(dst_route) - 1):
                    cust_j = dst_route[pos_j]
                    new_src = src_route[:pos_i] + [cust_j] + src_route[pos_i+1:]
                    new_dst = dst_route[:pos_j] + [cust_i] + dst_route[pos_j+1:]
                    new_dists = [route_dist(r) for ri, r in enumerate(routes) if ri not in (longest_idx, dst_idx)]
                    new_dist_src = route_dist(new_src)
                    new_dist_dst = route_dist(new_dst)
                    new_max = max([new_dist_src, new_dist_dst] + new_dists)
                    if new_max < compute_max() - 1e-9:
                        routes[longest_idx] = new_src
                        routes[dst_idx] = new_dst
                        if new_max < best_max:
                            best_max = new_max
                            best_routes = copy_routes()
                            report_best_vrp(best_routes)
                        return True
        return False

    operators = [op_2opt, op_relocate, op_swap]
    weights = [1.0, 1.0, 1.0]
    num_ops = len(operators)
    successes = [0.0 for _ in range(num_ops)]
    trials = [0 for _ in range(num_ops)]

    max_iter = 100 * n
    for iteration in range(max_iter):
        total_w = sum(weights)
        r = random.random() * total_w
        cum = 0.0
        op_idx = -1
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                op_idx = i
                break
        improved = operators[op_idx]()
        trials[op_idx] += 1
        if improved:
            successes[op_idx] += 1.0
        if (iteration + 1) % 50 == 0:
            for i in range(num_ops):
                if trials[i] > 0:
                    rate = successes[i] / trials[i]
                else:
                    rate = 0.5
                weights[i] = max(0.1, weights[i] * (0.9 + 0.1 * (rate - 0.33) / 0.33 if rate > 0.33 else 1.0))
            successes = [0.0 for _ in range(num_ops)]
            trials = [0 for _ in range(num_ops)]

    return best_routes