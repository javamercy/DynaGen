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

    def compute_total(routes):
        return sum(route_dist(r) for r in routes)

    def copy_routes(routes):
        return [list(r) for r in routes]

    # Greedy construction: insert each customer to minimize resulting max distance, deterministic
    def construct_greedy():
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
                            best_pairs = [(cust, ri, pos)]
                        elif new_max == best_max:
                            best_pairs.append((cust, ri, pos))
            if not best_pairs:
                break
            # deterministic: smallest customer, then route index, then position
            best_pairs.sort(key=lambda x: (x[0], x[1], x[2]))
            best_cust, best_ri, best_pos = best_pairs[0]
            routes[best_ri].insert(best_pos, best_cust)
            remaining.remove(best_cust)
        return routes

    # initial solution
    best_routes = construct_greedy()
    best_max = compute_max(best_routes)
    report_best_vrp(best_routes)

    # Simulated annealing parameters
    max_iter = 5000
    T0 = best_max * 0.5
    T = T0
    current_routes = copy_routes(best_routes)
    current_max = best_max
    current_total = compute_total(current_routes)

    for iteration in range(max_iter):
        T = T0 * (1 - iteration / max_iter)
        # choose move type: 0 relocate, 1 swap, 2 2-opt
        move_type = random.randint(0, 2)
        if move_type == 0:
            # relocate from longest route
            dists = [route_dist(r) for r in current_routes]
            longest_idx = max(range(truck_count), key=lambda i: (dists[i], i))
            src_route = current_routes[longest_idx]
            if len(src_route) <= 2:
                continue
            cand_positions = list(range(1, len(src_route)-1))
            if not cand_positions:
                continue
            pos_i = random.choice(cand_positions)
            cust = src_route[pos_i]
            dst_idx = random.randint(0, truck_count-1)
            while dst_idx == longest_idx:
                dst_idx = random.randint(0, truck_count-1)
            dst_route = current_routes[dst_idx]
            pos_j = random.randint(1, len(dst_route)-1)
            # apply move
            new_routes = copy_routes(current_routes)
            new_routes[longest_idx].pop(pos_i)
            new_routes[dst_idx].insert(pos_j, cust)
        elif move_type == 1:
            # swap between two different routes
            if truck_count < 2:
                continue
            ri = random.randint(0, truck_count-1)
            rj = random.randint(0, truck_count-1)
            while rj == ri:
                rj = random.randint(0, truck_count-1)
            route_i = current_routes[ri]
            route_j = current_routes[rj]
            if len(route_i) <= 2 or len(route_j) <= 2:
                continue
            pos_i = random.randint(1, len(route_i)-2)
            pos_j = random.randint(1, len(route_j)-2)
            cust_i = route_i[pos_i]
            cust_j = route_j[pos_j]
            new_routes = copy_routes(current_routes)
            new_routes[ri][pos_i] = cust_j
            new_routes[rj][pos_j] = cust_i
        else:
            # 2-opt on a random route
            ri = random.randint(0, truck_count-1)
            route = current_routes[ri]
            if len(route) <= 3:
                continue
            i = random.randint(1, len(route)-3)
            j = random.randint(i+1, len(route)-2)
            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
            new_routes = copy_routes(current_routes)
            new_routes[ri] = new_route

        new_max = compute_max(new_routes)
        new_total = compute_total(new_routes)
        if new_max < current_max or (new_max == current_max and new_total < current_total):
            accept = True
        elif T > 1e-12:
            delta = (new_max - current_max) / (current_max + 1e-12)
            prob = math.exp(-delta / T)
            accept = random.random() < prob
        else:
            accept = False

        if accept:
            current_routes = new_routes
            current_max = new_max
            current_total = new_total
            if current_max < best_max or (current_max == best_max and current_total < compute_total(best_routes)):
                best_routes = copy_routes(current_routes)
                best_max = current_max
                report_best_vrp(best_routes)

    return best_routes