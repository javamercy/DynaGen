import numpy as np
import random

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
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_dist(routes):
        return max(route_dist(r) for r in routes)

    def compute_insertion_cost(route, cust):
        best_delta = float('inf')
        best_pos = -1
        for pos in range(1, len(route)):
            prev = route[pos-1]
            nex = route[pos]
            delta = distance_matrix[prev, cust] + distance_matrix[cust, nex] - distance_matrix[prev, nex]
            if delta < best_delta:
                best_delta = delta
                best_pos = pos
        return best_delta, best_pos

    def regret_insertion(remaining, current_routes):
        routes = [r[:] for r in current_routes]
        unassigned = set(remaining)
        while unassigned:
            best_regret = -float('inf')
            best_cust = None
            best_route_idx = None
            best_pos = None
            for cust in unassigned:
                deltas = []
                positions = []
                for idx, route in enumerate(routes):
                    delta, pos = compute_insertion_cost(route, cust)
                    deltas.append(delta)
                    positions.append(pos)
                deltas_sorted = sorted(deltas)
                if len(deltas_sorted) >= 2:
                    regret = deltas_sorted[1] - deltas_sorted[0]
                else:
                    regret = deltas_sorted[0]
                # Choose best regret; tie-break by node index
                if regret > best_regret or (regret == best_regret and (best_cust is None or cust < best_cust)):
                    best_regret = regret
                    best_cust = cust
                    best_route_idx = deltas.index(deltas_sorted[0])
                    best_pos = positions[best_route_idx]
            routes[best_route_idx].insert(best_pos, best_cust)
            unassigned.remove(best_cust)
            report_best_vrp(routes)
        return routes

    # Seed selection: farthest-first from depot
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

    # Local search functions
    def local_search(routes):
        max_iter = min(50, n * truck_count)
        for _ in range(max_iter):
            improved = False
            current_max = max_dist(routes)

            # Relocate
            best_move = None
            best_new_max = current_max
            for i in range(truck_count):
                route_i = routes[i]
                if len(route_i) <= 2:
                    continue
                for pos in range(1, len(route_i)-1):
                    cust = route_i[pos]
                    for j in range(truck_count):
                        if j == i:
                            continue
                        route_j = routes[j]
                        _, ins_pos = compute_insertion_cost(route_j, cust)
                        new_i = route_i[:pos] + route_i[pos+1:]
                        new_j = route_j[:ins_pos] + [cust] + route_j[ins_pos:]
                        dist_i = route_dist(new_i)
                        dist_j = route_dist(new_j)
                        other_max = 0.0
                        for k, r in enumerate(routes):
                            if k == i or k == j:
                                continue
                            other_max = max(other_max, route_dist(r))
                        new_max = max(other_max, dist_i, dist_j)
                        if new_max < best_new_max or (new_max == best_new_max and (best_move is None or (i, pos, j) < (best_move[1], best_move[2], best_move[3]))):
                            best_new_max = new_max
                            best_move = ('relocate', i, pos, j, ins_pos)
            if best_move is not None:
                _, i, pos, j, ins_pos = best_move
                cust = routes[i][pos]
                routes[i] = routes[i][:pos] + routes[i][pos+1:]
                routes[j] = routes[j][:ins_pos] + [cust] + routes[j][ins_pos:]
                improved = True
                report_best_vrp(routes)
                continue

            # Swap
            best_move = None
            best_new_max = max_dist(routes)
            for i in range(truck_count):
                route_i = routes[i]
                if len(route_i) <= 2:
                    continue
                for pos_i in range(1, len(route_i)-1):
                    cust_i = route_i[pos_i]
                    for j in range(i+1, truck_count):
                        route_j = routes[j]
                        if len(route_j) <= 2:
                            continue
                        for pos_j in range(1, len(route_j)-1):
                            cust_j = route_j[pos_j]
                            new_i = route_i[:]
                            new_i[pos_i] = cust_j
                            new_j = route_j[:]
                            new_j[pos_j] = cust_i
                            dist_i = route_dist(new_i)
                            dist_j = route_dist(new_j)
                            other_max = 0.0
                            for k, r in enumerate(routes):
                                if k == i or k == j:
                                    continue
                                other_max = max(other_max, route_dist(r))
                            new_max = max(other_max, dist_i, dist_j)
                            if new_max < best_new_max or (new_max == best_new_max and (best_move is None or (i, pos_i, j, pos_j) < (best_move[1], best_move[2], best_move[3], best_move[4]))):
                                best_new_max = new_max
                                best_move = ('swap', i, pos_i, j, pos_j)
            if best_move is not None:
                _, i, pos_i, j, pos_j = best_move
                cust_i = routes[i][pos_i]
                cust_j = routes[j][pos_j]
                routes[i][pos_i] = cust_j
                routes[j][pos_j] = cust_i
                improved = True
                report_best_vrp(routes)
                continue

            # 2-opt
            best_move = None
            best_new_max = max_dist(routes)
            for i in range(truck_count):
                route_i = routes[i]
                if len(route_i) <= 3:
                    continue
                for a in range(1, len(route_i)-2):
                    for b in range(a+1, len(route_i)-1):
                        new_i = route_i[:a] + route_i[a:b+1][::-1] + route_i[b+1:]
                        dist_i = route_dist(new_i)
                        other_max = 0.0
                        for k, r in enumerate(routes):
                            if k == i:
                                continue
                            other_max = max(other_max, route_dist(r))
                        new_max = max(other_max, dist_i)
                        if new_max < best_new_max or (new_max == best_new_max and (best_move is None or (i, a, b) < (best_move[1], best_move[2], best_move[3]))):
                            best_new_max = new_max
                            best_move = ('2opt', i, a, b)
            if best_move is not None:
                _, i, a, b = best_move
                routes[i] = routes[i][:a] + routes[i][a:b+1][::-1] + routes[i][b+1:]
                improved = True
                report_best_vrp(routes)
                continue

            # Cross-2-opt*
            best_move = None
            best_new_max = max_dist(routes)
            for i in range(truck_count):
                route_i = routes[i]
                if len(route_i) <= 2:
                    continue
                for j in range(i+1, truck_count):
                    route_j = routes[j]
                    if len(route_j) <= 2:
                        continue
                    for cut_i in range(0, len(route_i)-1):
                        for cut_j in range(0, len(route_j)-1):
                            new_i = route_i[:cut_i+1] + route_j[cut_j+1:]
                            new_j = route_j[:cut_j+1] + route_i[cut_i+1:]
                            if len(new_i) < 2 or len(new_j) < 2:
                                continue
                            dist_i = route_dist(new_i)
                            dist_j = route_dist(new_j)
                            other_max = 0.0
                            for k, r in enumerate(routes):
                                if k == i or k == j:
                                    continue
                                other_max = max(other_max, route_dist(r))
                            new_max = max(other_max, dist_i, dist_j)
                            if new_max < best_new_max or (new_max == best_new_max and (best_move is None or (i, j, cut_i, cut_j) < (best_move[1], best_move[2], best_move[3], best_move[4]))):
                                best_new_max = new_max
                                best_move = ('cross', i, j, cut_i, cut_j)
            if best_move is not None:
                _, i, j, cut_i, cut_j = best_move
                orig_i = routes[i][:]
                orig_j = routes[j][:]
                routes[i] = orig_i[:cut_i+1] + orig_j[cut_j+1:]
                routes[j] = orig_j[:cut_j+1] + orig_i[cut_i+1:]
                improved = True
                report_best_vrp(routes)
                continue

            if not improved:
                break
        return routes

    # Initial construction
    initial_routes = [[0, s, 0] for s in seeds]
    remaining = [c for c in customers if c not in seeds]
    routes = regret_insertion(remaining, initial_routes)
    routes = local_search(routes)
    curr_max = max_dist(routes)
    if curr_max < best_max:
        best_max = curr_max
        best_routes = [r[:] for r in routes]

    # Restart loop with biased removal and regret reinsertion
    max_restarts = 3
    for restart in range(max_restarts):
        random.seed(restart)
        # Compute average route distance
        dists = [route_dist(r) for r in routes]
        avg_dist = sum(dists) / len(dists) if dists else 0
        to_remove = set()
        # Bias removal: from routes with distance > avg, remove a higher proportion
        for idx, r in enumerate(routes):
            if len(r) <= 2:
                continue
            prop = 0.3 if dists[idx] > avg_dist else 0.1
            num_remove = max(1, int(len(r[1:-1]) * prop))
            # Remove random customers from those routes
            candidates = r[1:-1]
            if candidates:
                removed = random.sample(candidates, min(num_remove, len(candidates)))
                to_remove.update(removed)
        # Remove the customers from routes
        new_routes = []
        for r in routes:
            new_route = [0]
            for node in r[1:-1]:
                if node not in to_remove:
                    new_route.append(node)
            new_route.append(0)
            new_routes.append(new_route)
        # Reinsert removed customers using regret insertion
        remaining = list(to_remove)
        new_routes = regret_insertion(remaining, new_routes)
        # Local search
        new_routes = local_search(new_routes)
        new_max = max_dist(new_routes)
        if new_max < best_max:
            best_max = new_max
            best_routes = [r[:] for r in new_routes]
        routes = new_routes  # always accept for diversity

    if best_routes is None:
        best_routes = routes
    while len(best_routes) < truck_count:
        best_routes.append([0, 0])
    report_best_vrp(best_routes)
    return best_routes