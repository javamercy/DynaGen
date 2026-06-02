import numpy as np
import random
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n - 1:
        routes = [[0, i, 0] for i in range(1, n)]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes

    def route_dist(route):
        d = 0.0
        for i in range(len(route) - 1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_dist(routes):
        return max(route_dist(r) for r in routes)

    def regret_insertion(remaining, routes, regret_k=2):
        rem = list(remaining)
        while rem:
            best_cust = None
            best_route_idx = -1
            best_pos = -1
            best_regret = -float('inf')
            for cust in rem:
                # Compute best and second best insertion cost for each route
                deltas = []
                for idx, route in enumerate(routes):
                    best_delta = float('inf')
                    best_local_pos = -1
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nex = route[pos]
                        delta = distance_matrix[prev, cust] + distance_matrix[cust, nex] - distance_matrix[prev, nex]
                        if delta < best_delta:
                            best_delta = delta
                            best_local_pos = pos
                    deltas.append((best_delta, best_local_pos, idx))
                deltas.sort(key=lambda x: x[0])
                # Regret = difference between second best and best
                if len(deltas) >= 2:
                    regret = deltas[1][0] - deltas[0][0]
                else:
                    regret = deltas[0][0]
                if regret > best_regret:
                    best_regret = regret
                    best_cust = cust
                    best_route_idx = deltas[0][2]
                    best_pos = deltas[0][1]
            routes[best_route_idx].insert(best_pos, best_cust)
            rem.remove(best_cust)
            report_best_vrp(routes)
        return routes

    def construct_initial(seeds):
        routes = [[0, s, 0] for s in seeds]
        remaining = [c for c in customers if c not in seeds]
        remaining.sort(key=lambda c: -distance_matrix[0, c])
        return regret_insertion(remaining, routes, regret_k=2)

    # Seed selection: farthest from depot and each other
    seeds = []
    first_seed = max(range(1, n), key=lambda i: (distance_matrix[0, i], -i))
    seeds.append(first_seed)
    for _ in range(1, truck_count):
        best_min_dist = -1
        best_node = None
        for node in range(1, n):
            if node in seeds:
                continue
            min_dist = min(distance_matrix[node, s] for s in seeds)
            if min_dist > best_min_dist or (min_dist == best_min_dist and (best_node is None or node < best_node)):
                best_min_dist = min_dist
                best_node = node
        if best_node is None:
            break
        seeds.append(best_node)

    best_routes = None
    best_max = float('inf')

    def local_search_sa(routes, initial_temp, alpha):
        current_routes = [r[:] for r in routes]
        current_max = max_dist(current_routes)
        best_routes_local = [r[:] for r in current_routes]
        best_max_local = current_max

        max_iter = min(100, n * truck_count * 2)
        temp = initial_temp
        for iteration in range(max_iter):
            # Generate neighbor: relocate, swap, or 2-opt (random choice)
            neighbor = [r[:] for r in current_routes]
            op = random.randint(0, 2)
            if op == 0:  # relocate
                i = random.randint(0, truck_count-1)
                if len(neighbor[i]) <= 2:
                    continue
                pos = random.randint(1, len(neighbor[i])-2)
                cust = neighbor[i][pos]
                j = random.randint(0, truck_count-1)
                if j == i:
                    continue
                ins_pos = random.randint(1, len(neighbor[j])-1)
                neighbor[i].pop(pos)
                neighbor[j].insert(ins_pos, cust)
            elif op == 1:  # swap
                i = random.randint(0, truck_count-1)
                if len(neighbor[i]) <= 2:
                    continue
                pos_i = random.randint(1, len(neighbor[i])-2)
                j = random.randint(0, truck_count-1)
                if j == i or len(neighbor[j]) <= 2:
                    continue
                pos_j = random.randint(1, len(neighbor[j])-2)
                cust_i = neighbor[i][pos_i]
                cust_j = neighbor[j][pos_j]
                neighbor[i][pos_i] = cust_j
                neighbor[j][pos_j] = cust_i
            else:  # 2-opt intra-route
                i = random.randint(0, truck_count-1)
                if len(neighbor[i]) <= 3:
                    continue
                a = random.randint(1, len(neighbor[i])-3)
                b = random.randint(a+1, len(neighbor[i])-2)
                neighbor[i] = neighbor[i][:a] + neighbor[i][a:b+1][::-1] + neighbor[i][b+1:]

            new_max = max_dist(neighbor)
            delta = new_max - current_max
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_routes = neighbor
                current_max = new_max
                if new_max < best_max_local:
                    best_max_local = new_max
                    best_routes_local = [r[:] for r in current_routes]
                    report_best_vrp(best_routes_local)
            temp *= alpha

        return best_routes_local, best_max_local

    # Initial solution
    routes = construct_initial(seeds)
    # Initial SA parameters
    initial_max = max_dist(routes)
    initial_temp = 0.1 * initial_max if initial_max > 0 else 1.0
    alpha = 0.98
    routes, curr_max = local_search_sa(routes, initial_temp, alpha)
    best_max = curr_max
    best_routes = [r[:] for r in routes]

    # Restarts with perturbation
    max_restarts = 5
    random.seed(12345)
    for restart in range(max_restarts):
        # Remove random subset
        all_customers = list(range(1, n))
        remove_count = max(1, n // 10)
        to_remove = set(random.sample(all_customers, remove_count))
        new_routes = []
        for r in best_routes:
            new_route = [0]
            for node in r[1:-1]:
                if node not in to_remove:
                    new_route.append(node)
            new_route.append(0)
            new_routes.append(new_route)
        remaining = list(to_remove)
        new_routes = regret_insertion(remaining, new_routes)
        # Run SA on perturbed solution
        new_routes, new_max = local_search_sa(new_routes, initial_temp, alpha)
        if new_max < best_max:
            best_max = new_max
            best_routes = [r[:] for r in new_routes]
            report_best_vrp(best_routes)

    while len(best_routes) < truck_count:
        best_routes.append([0, 0])
    report_best_vrp(best_routes)
    return best_routes