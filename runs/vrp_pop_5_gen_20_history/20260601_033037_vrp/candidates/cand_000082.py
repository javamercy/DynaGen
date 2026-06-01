import numpy as np
import collections

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    best_routes = None
    best_max = float('inf')

    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        new_max = max(route_distance(r) for r in routes)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [list(r) for r in routes]

    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    # Construction: farthest-first seeds + regret-2 assignment + cheapest insertion
    seeds = []
    seed0 = max(customers, key=lambda x: distance_matrix[0, x])
    seeds.append(seed0)
    while len(seeds) < truck_count:
        best_cust = None
        best_min_dist = -1.0
        for c in customers:
            if c in seeds:
                continue
            min_dist = min(distance_matrix[c, s] for s in seeds)
            if min_dist > best_min_dist or (min_dist == best_min_dist and (best_cust is None or c < best_cust)):
                best_min_dist = min_dist
                best_cust = c
        if best_cust is not None:
            seeds.append(best_cust)
        else:
            break

    unassigned = [c for c in customers if c not in seeds]
    clusters = [[] for _ in range(truck_count)]
    while unassigned:
        best_list = []
        for c in unassigned:
            dists = [distance_matrix[c, s] for s in seeds]
            sorted_dists = sorted(dists)
            regret = sorted_dists[1] - sorted_dists[0] if len(sorted_dists) > 1 else 0.0
            nearest_idx = dists.index(sorted_dists[0])
            best_list.append((regret, c, nearest_idx))
        best_list.sort(key=lambda x: (-x[0], x[1]))
        regret, c, seed_idx = best_list[0]
        clusters[seed_idx].append(c)
        unassigned.remove(c)
    for i, s in enumerate(seeds):
        clusters[i].append(s)

    def build_routes_from_clusters(clusters):
        routes = []
        for cl in clusters:
            if not cl:
                routes.append([0, 0])
            else:
                route = [0, 0]
                remaining = list(cl)
                while remaining:
                    best_cust = None
                    best_pos = None
                    best_cost = float('inf')
                    for c in remaining:
                        for pos in range(1, len(route)):
                            delta = distance_matrix[route[pos-1], c] + distance_matrix[c, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                            if delta < best_cost - 1e-12 or (abs(delta - best_cost) < 1e-12 and (best_cust is None or c < best_cust)):
                                best_cost = delta
                                best_cust = c
                                best_pos = pos
                    route.insert(best_pos, best_cust)
                    remaining.remove(best_cust)
                routes.append(route)
        return routes

    routes = build_routes_from_clusters(clusters)
    report_best_vrp(routes)

    # Tabu search parameters
    max_iter = min(500, n * truck_count)
    tabu_tenure = max(5, n // 10)
    tabu_list = collections.deque(maxlen=tabu_tenure)
    stagnation_limit = 15
    stagnation = 0
    prev_best = best_max

    for iteration in range(max_iter):
        improved = False
        dists = [route_distance(r) for r in routes]
        longest_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
        interior_long = routes[longest_idx][1:-1]
        if not interior_long:
            break

        best_move = None
        best_new_max = float('inf')

        # Relocate moves from longest route
        for cust in interior_long:
            for other_idx in range(truck_count):
                if other_idx == longest_idx:
                    continue
                move_key = (cust, longest_idx, other_idx)
                if move_key in tabu_list:
                    continue
                other_route = routes[other_idx]
                for pos in range(1, len(other_route)):
                    new_routes = [list(r) for r in routes]
                    new_routes[longest_idx].remove(cust)
                    new_routes[other_idx].insert(pos, cust)
                    new_max = max(route_distance(r) for r in new_routes)
                    if new_max < best_new_max - 1e-12:
                        best_new_max = new_max
                        best_move = ('relocate', cust, longest_idx, other_idx, pos, new_routes)
                    elif abs(new_max - best_new_max) < 1e-12:
                        if cust < (best_move[1] if best_move else float('inf')):
                            best_new_max = new_max
                            best_move = ('relocate', cust, longest_idx, other_idx, pos, new_routes)

        # Swap moves
        for other_idx in range(truck_count):
            if other_idx == longest_idx:
                continue
            other_interior = routes[other_idx][1:-1]
            if not other_interior:
                continue
            for cust_max in interior_long:
                for cust_other in other_interior:
                    move_key = (cust_max, cust_other)
                    rev_key = (cust_other, cust_max)
                    if move_key in tabu_list or rev_key in tabu_list:
                        continue
                    new_routes = [list(r) for r in routes]
                    idx_max = new_routes[longest_idx].index(cust_max)
                    idx_other = new_routes[other_idx].index(cust_other)
                    new_routes[longest_idx][idx_max] = cust_other
                    new_routes[other_idx][idx_other] = cust_max
                    new_max = max(route_distance(r) for r in new_routes)
                    if new_max < best_new_max - 1e-12:
                        best_new_max = new_max
                        best_move = ('swap', cust_max, longest_idx, cust_other, other_idx, new_routes)
                    elif abs(new_max - best_new_max) < 1e-12:
                        if cust_max < (best_move[1] if best_move else float('inf')):
                            best_new_max = new_max
                            best_move = ('swap', cust_max, longest_idx, cust_other, other_idx, new_routes)

        # Apply best move if non-tabu or aspiration
        if best_move is not None:
            if best_new_max < best_max - 1e-12:  # aspiration
                if best_move[0] == 'relocate':
                    _, cust, from_idx, to_idx, pos, new_routes = best_move
                    tabu_list.append((cust, from_idx, to_idx))
                else:
                    _, cust_max, max_idx, cust_other, other_idx, new_routes = best_move
                    tabu_list.append((cust_max, cust_other))
                routes = new_routes
                report_best_vrp(routes)
                improved = True
            else:
                # best_move is guaranteed non-tabu
                if best_move[0] == 'relocate':
                    _, cust, from_idx, to_idx, pos, new_routes = best_move
                    tabu_list.append((cust, from_idx, to_idx))
                else:
                    _, cust_max, max_idx, cust_other, other_idx, new_routes = best_move
                    tabu_list.append((cust_max, cust_other))
                routes = new_routes
                improved = True

        # If no inter-route move, try intra-route 2-opt and Or-opt
        if not improved:
            for idx in range(truck_count):
                route = routes[idx]
                if len(route) <= 3:
                    continue
                best_route = list(route)
                best_dist = route_distance(route)
                # 2-opt
                found = False
                for a in range(1, len(route)-2):
                    for b in range(a+1, len(route)-1):
                        new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                        new_dist = route_distance(new_route)
                        if new_dist < best_dist - 1e-12:
                            best_dist = new_dist
                            best_route = new_route
                            found = True
                            break
                    if found:
                        break
                if not found:
                    # Or-opt (segment lengths 1,2,3)
                    for seg_len in range(1, min(4, len(route)-1)):
                        for start in range(1, len(route)-seg_len):
                            end = start + seg_len - 1
                            segment = route[start:end+1]
                            remaining = route[:start] + route[end+1:]
                            for pos in range(1, len(remaining)):
                                new_route = remaining[:pos] + segment + remaining[pos:]
                                if new_route[0] != 0 or new_route[-1] != 0:
                                    continue
                                d = route_distance(new_route)
                                if d < best_dist - 1e-12:
                                    best_dist = d
                                    best_route = new_route
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                if found:
                    routes[idx] = best_route
                    new_max = max(route_distance(r) for r in routes)
                    if new_max < best_max - 1e-12:
                        report_best_vrp(routes)
                    improved = True
                    break

        # If still no improvement, ruin-recreate
        if not improved:
            if len(interior_long) > 2:
                remove_count = max(1, len(interior_long) // 10)
                to_remove = interior_long[:remove_count]
                new_routes = [list(r) for r in routes]
                removed = []
                for cust in to_remove:
                    new_routes[longest_idx].remove(cust)
                    removed.append(cust)
                # Reinsert cheapest
                for cust in removed:
                    best_route_idx = None
                    best_pos = None
                    best_cost = float('inf')
                    for r_idx in range(truck_count):
                        r = new_routes[r_idx]
                        for pos in range(1, len(r)):
                            delta = distance_matrix[r[pos-1], cust] + distance_matrix[cust, r[pos]] - distance_matrix[r[pos-1], r[pos]]
                            if delta < best_cost - 1e-12 or (abs(delta - best_cost) < 1e-12 and (best_route_idx is None or r_idx < best_route_idx or (r_idx == best_route_idx and pos < best_pos))):
                                best_cost = delta
                                best_route_idx = r_idx
                                best_pos = pos
                    new_routes[best_route_idx].insert(best_pos, cust)
                new_max = max(route_distance(r) for r in new_routes)
                if new_max < best_max - 1e-12:
                    routes = new_routes
                    report_best_vrp(routes)
                    improved = True
        
        # Update stagnation
        curr_best = best_max
        if curr_best < prev_best - 1e-12:
            stagnation = 0
        else:
            stagnation += 1
        prev_best = curr_best
        if stagnation >= stagnation_limit:
            break

        if not improved:
            break

    final_routes = best_routes if best_routes is not None else routes
    while len(final_routes) < truck_count:
        final_routes.append([0, 0])
    return final_routes