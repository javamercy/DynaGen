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
        report_best_vrp(routes)
        return routes

    def route_dist(route):
        return sum(distance_matrix[route[k]][route[k+1]] for k in range(len(route)-1))

    def compute_max(routes):
        return max(route_dist(r) for r in routes)

    def copy_routes(routes):
        return [list(r) for r in routes]

    def construct_solution(randomized, rcl_fraction):
        routes = [[0, 0] for _ in range(truck_count)]
        remaining = set(customers)
        while remaining:
            best_max = math.inf
            best_pairs = []
            for cust in remaining:
                for ri, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_dist = route_dist(new_route)
                        other_dists = [route_dist(r) for rj, r in enumerate(routes) if rj != ri]
                        new_max = max(new_dist, *other_dists)
                        if new_max < best_max:
                            best_max = new_max
                            best_pairs = [(cust, ri, pos, new_max)]
                        elif new_max == best_max:
                            best_pairs.append((cust, ri, pos, new_max))
            if not best_pairs:
                break
            if not randomized:
                best_pairs.sort(key=lambda x: (x[0], x[1], x[2]))
                best_cust, best_ri, best_pos, _ = best_pairs[0]
            else:
                rcl_size = max(1, int(len(best_pairs) * rcl_fraction))
                selected = random.choice(best_pairs[:rcl_size])
                best_cust, best_ri, best_pos, _ = selected
            routes[best_ri].insert(best_pos, best_cust)
            remaining.remove(best_cust)
        for cust in remaining:
            best_max = math.inf
            best_ri = -1
            best_pos = -1
            for ri, route in enumerate(routes):
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_dist = route_dist(new_route)
                    other_dists = [route_dist(r) for rj, r in enumerate(routes) if rj != ri]
                    new_max = max(new_dist, *other_dists)
                    if new_max < best_max:
                        best_max = new_max
                        best_ri = ri
                        best_pos = pos
            routes[best_ri].insert(best_pos, cust)
        return routes

    def apply_vnd(routes):
        improved = True
        while improved:
            improved = False
            dists = [route_dist(r) for r in routes]
            max_dist = max(dists)
            avg_dist = sum(dists) / truck_count if truck_count else 0
            imbalance_threshold = 1.2 * avg_dist
            # Prioritize relocate if imbalance
            if max_dist > imbalance_threshold:
                longest_idx = max(range(truck_count), key=lambda i: dists[i])
                shortest_idx = min(range(truck_count), key=lambda i: dists[i])
                src_route = routes[longest_idx]
                if len(src_route) > 2:
                    for pos_i in range(1, len(src_route) - 1):
                        cust = src_route[pos_i]
                        dst_route = routes[shortest_idx]
                        for pos_j in range(1, len(dst_route)):
                            new_src = src_route[:pos_i] + src_route[pos_i+1:]
                            new_dst = dst_route[:pos_j] + [cust] + dst_route[pos_j:]
                            new_dists = [route_dist(r) for ri, r in enumerate(routes) if ri not in (longest_idx, shortest_idx)]
                            new_max = max([route_dist(new_src), route_dist(new_dst)] + new_dists)
                            if new_max < compute_max(routes) - 1e-9:
                                routes[longest_idx] = new_src
                                routes[shortest_idx] = new_dst
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        continue
            # 2-opt (intra-route)
            for ri, route in enumerate(routes):
                if len(route) <= 3:
                    continue
                best_local_dist = route_dist(route)
                best_local_route = route[:]
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
                    break
            if improved:
                continue
            # Relocate general (if not already done)
            if max_dist <= imbalance_threshold:
                dists = [route_dist(r) for r in routes]
                longest_idx = max(range(truck_count), key=lambda i: dists[i])
                src_route = routes[longest_idx]
                if len(src_route) > 2:
                    for pos_i in range(1, len(src_route) - 1):
                        cust = src_route[pos_i]
                        for dst_idx in range(truck_count):
                            if dst_idx == longest_idx:
                                continue
                            dst_route = routes[dst_idx]
                            for pos_j in range(1, len(dst_route)):
                                new_src = src_route[:pos_i] + src_route[pos_i+1:]
                                new_dst = dst_route[:pos_j] + [cust] + dst_route[pos_j:]
                                new_dists = [route_dist(r) for ri, r in enumerate(routes) if ri not in (longest_idx, dst_idx)]
                                new_max = max([route_dist(new_src), route_dist(new_dst)] + new_dists)
                                if new_max < compute_max(routes) - 1e-9:
                                    routes[longest_idx] = new_src
                                    routes[dst_idx] = new_dst
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
            if improved:
                continue
            # Swap (inter-route)
            dists = [route_dist(r) for r in routes]
            sorted_indices = sorted(range(truck_count), key=lambda i: -dists[i])
            for i in range(truck_count):
                for j in range(i+1, truck_count):
                    ri = sorted_indices[i]
                    rj = sorted_indices[j]
                    route_i = routes[ri]
                    route_j = routes[rj]
                    if len(route_i) <= 2 or len(route_j) <= 2:
                        continue
                    for pos_i in range(1, len(route_i) - 1):
                        cust_i = route_i[pos_i]
                        for pos_j in range(1, len(route_j) - 1):
                            cust_j = route_j[pos_j]
                            new_i = route_i[:pos_i] + [cust_j] + route_i[pos_i+1:]
                            new_j = route_j[:pos_j] + [cust_i] + route_j[pos_j+1:]
                            new_dists = [route_dist(r) for ri2, r in enumerate(routes) if ri2 not in (ri, rj)]
                            new_max = max([route_dist(new_i), route_dist(new_j)] + new_dists)
                            if new_max < compute_max(routes) - 1e-9:
                                routes[ri] = new_i
                                routes[rj] = new_j
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    # Initial deterministic construction
    best_routes = construct_solution(randomized=False, rcl_fraction=0.2)
    best_max = compute_max(best_routes)
    report_best_vrp(best_routes)

    # Apply VND to initial solution
    routes_improved = apply_vnd(copy_routes(best_routes))
    new_max = compute_max(routes_improved)
    if new_max < best_max:
        best_routes = copy_routes(routes_improved)
        best_max = new_max
        report_best_vrp(best_routes)

    # GRASP restarts with adaptive RCL
    max_restarts = 10 + n // 10
    for restart in range(max_restarts):
        # Adaptive RCL fraction: decay from 0.5 to 0.05
        rcl_fraction = 0.5 - 0.45 * (restart / max_restarts)
        new_routes = construct_solution(randomized=True, rcl_fraction=rcl_fraction)
        new_routes = apply_vnd(new_routes)
        new_max = compute_max(new_routes)
        if new_max < best_max:
            best_routes = copy_routes(new_routes)
            best_max = new_max
            report_best_vrp(best_routes)

    return best_routes