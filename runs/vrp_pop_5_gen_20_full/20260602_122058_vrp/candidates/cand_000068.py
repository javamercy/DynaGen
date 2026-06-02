import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n - 1:
        routes = [[0, i, 0] for i in range(1, n)]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_dist(route):
        d = 0.0
        for i in range(len(route) - 1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_dist(routes):
        return max(route_dist(r) for r in routes)

    # Seed selection: farthest-first from depot (tie-break smallest index)
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

    def regret2_insertion(route_list, unassigned):
        while unassigned:
            best_regret = -float('inf')
            best_cust = None
            best_idx = -1
            best_pos = -1
            for cust in unassigned:
                costs = []
                for idx, route in enumerate(route_list):
                    best_delta = float('inf')
                    best_pos_local = -1
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nex = route[pos]
                        delta = distance_matrix[prev, cust] + distance_matrix[cust, nex] - distance_matrix[prev, nex]
                        if delta < best_delta:
                            best_delta = delta
                            best_pos_local = pos
                    costs.append((best_delta, best_pos_local, idx))
                costs.sort(key=lambda x: (x[0], x[2]))
                if len(costs) >= 2:
                    regret = costs[1][0] - costs[0][0]
                else:
                    regret = costs[0][0]
                if regret > best_regret or (regret == best_regret and (best_cust is None or cust < best_cust)):
                    best_regret = regret
                    best_cust = cust
                    best_idx = costs[0][2]
                    best_pos = costs[0][1]
            route_list[best_idx].insert(best_pos, best_cust)
            unassigned.remove(best_cust)

    def construct_initial(seeds):
        routes = [[0, s, 0] for s in seeds]
        remaining = [c for c in customers if c not in seeds]
        # Insert remaining with regret-2
        regret2_insertion(routes, remaining)
        return routes

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
                for pos in range(1, len(route_i) - 1):
                    cust = route_i[pos]
                    for j in range(truck_count):
                        if j == i:
                            continue
                        route_j = routes[j]
                        best_delta = float('inf')
                        best_pos_j = -1
                        for ins_pos in range(1, len(route_j)):
                            prev = route_j[ins_pos - 1]
                            nex = route_j[ins_pos]
                            delta = distance_matrix[prev, cust] + distance_matrix[cust, nex] - distance_matrix[prev, nex]
                            if delta < best_delta:
                                best_delta = delta
                                best_pos_j = ins_pos
                        new_route_i = route_i[:pos] + route_i[pos+1:]
                        new_route_j = route_j[:best_pos_j] + [cust] + route_j[best_pos_j:]
                        dist_i = route_dist(new_route_i)
                        dist_j = route_dist(new_route_j)
                        other_max = 0.0
                        for k, r in enumerate(routes):
                            if k == i or k == j:
                                continue
                            other_max = max(other_max, route_dist(r))
                        new_max = max(other_max, dist_i, dist_j)
                        if new_max < best_new_max or (new_max == best_new_max and (i, pos, j, best_pos_j) < (best_move[1] if best_move else (float('inf'),))):
                            best_new_max = new_max
                            best_move = ('relocate', i, pos, j, best_pos_j)
            if best_move is not None:
                _, i, pos, j, ins_pos = best_move
                cust = routes[i][pos]
                routes[i] = routes[i][:pos] + routes[i][pos+1:]
                routes[j] = routes[j][:ins_pos] + [cust] + routes[j][ins_pos:]
                improved = True
                continue

            # Swap
            best_move = None
            best_new_max = max_dist(routes)
            for i in range(truck_count):
                route_i = routes[i]
                if len(route_i) <= 2:
                    continue
                for pos_i in range(1, len(route_i) - 1):
                    cust_i = route_i[pos_i]
                    for j in range(i + 1, truck_count):
                        route_j = routes[j]
                        if len(route_j) <= 2:
                            continue
                        for pos_j in range(1, len(route_j) - 1):
                            cust_j = route_j[pos_j]
                            new_route_i = route_i[:]
                            new_route_i[pos_i] = cust_j
                            new_route_j = route_j[:]
                            new_route_j[pos_j] = cust_i
                            dist_i = route_dist(new_route_i)
                            dist_j = route_dist(new_route_j)
                            other_max = 0.0
                            for k, r in enumerate(routes):
                                if k == i or k == j:
                                    continue
                                other_max = max(other_max, route_dist(r))
                            new_max = max(other_max, dist_i, dist_j)
                            if new_max < best_new_max or (new_max == best_new_max and (i, pos_i, j, pos_j) < (best_move[1] if best_move else (float('inf'),))):
                                best_new_max = new_max
                                best_move = ('swap', i, pos_i, j, pos_j)
            if best_move is not None:
                _, i, pos_i, j, pos_j = best_move
                cust_i = routes[i][pos_i]
                cust_j = routes[j][pos_j]
                routes[i][pos_i] = cust_j
                routes[j][pos_j] = cust_i
                improved = True
                continue

            # 2-opt
            best_move = None
            best_new_max = max_dist(routes)
            for i in range(truck_count):
                route_i = routes[i]
                if len(route_i) <= 3:
                    continue
                for a in range(1, len(route_i) - 2):
                    for b in range(a + 1, len(route_i) - 1):
                        new_route_i = route_i[:a] + route_i[a:b+1][::-1] + route_i[b+1:]
                        dist_i = route_dist(new_route_i)
                        other_max = 0.0
                        for k, r in enumerate(routes):
                            if k == i:
                                continue
                            other_max = max(other_max, route_dist(r))
                        new_max = max(other_max, dist_i)
                        if new_max < best_new_max or (new_max == best_new_max and (i, a, b) < (best_move[1] if best_move else (float('inf'),))):
                            best_new_max = new_max
                            best_move = ('2opt', i, a, b)
            if best_move is not None:
                _, i, a, b = best_move
                routes[i] = routes[i][:a] + routes[i][a:b+1][::-1] + routes[i][b+1:]
                improved = True
                continue

            # Cross-2-opt*
            best_move = None
            best_new_max = max_dist(routes)
            for i in range(truck_count):
                route_i = routes[i]
                if len(route_i) <= 2:
                    continue
                for j in range(i + 1, truck_count):
                    route_j = routes[j]
                    if len(route_j) <= 2:
                        continue
                    for cut_i in range(0, len(route_i) - 1):
                        for cut_j in range(0, len(route_j) - 1):
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
                            if new_max < best_new_max or (new_max == best_new_max and (i, j, cut_i, cut_j) < (best_move[1] if best_move else (float('inf'),))):
                                best_new_max = new_max
                                best_move = ('cross', i, j, cut_i, cut_j)
            if best_move is not None:
                _, i, j, cut_i, cut_j = best_move
                orig_i = routes[i][:]
                orig_j = routes[j][:]
                routes[i] = orig_i[:cut_i+1] + orig_j[cut_j+1:]
                routes[j] = orig_j[:cut_j+1] + orig_i[cut_i+1:]
                improved = True
                continue

            if not improved:
                break
        return routes

    # Build initial solution
    routes = construct_initial(seeds)
    routes = local_search(routes)
    best_routes = [r[:] for r in routes]
    best_max = max_dist(routes)

    # Record-to-record travel: accept solutions within threshold of best
    max_restarts = 20
    threshold = best_max * 0.5  # 50% deviation from best
    for restart in range(max_restarts):
        random.seed(restart)
        all_cust = list(range(1, n))
        # Remove 50% customers
        remove_count = max(1, len(all_cust) * 50 // 100)
        to_remove = set(random.sample(all_cust, remove_count))

        # Remove customers from current best routes? Use best_routes to start perturbation
        current_routes = [r[:] for r in best_routes]
        new_routes = []
        for r in current_routes:
            new_route = [0]
            for node in r[1:-1]:
                if node not in to_remove:
                    new_route.append(node)
            new_route.append(0)
            new_routes.append(new_route)

        # Shake remaining
        remaining = list(to_remove)
        random.shuffle(remaining)

        # Reinsert with regret-2
        regret2_insertion(new_routes, remaining)

        # Local search
        new_routes = local_search(new_routes)
        new_max = max_dist(new_routes)
        if new_max < best_max:
            best_max = new_max
            best_routes = [r[:] for r in new_routes]
        # Accept if within threshold of best (to escape local optima)
        if new_max <= best_max + threshold:
            routes = new_routes  # use for next perturbation
        # else keep current routes as best_routes for next perturbation

    # Ensure exactly truck_count routes
    while len(best_routes) < truck_count:
        best_routes.append([0, 0])
    return best_routes