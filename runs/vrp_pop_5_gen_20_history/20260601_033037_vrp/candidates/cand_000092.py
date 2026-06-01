import numpy as np
import heapq
import collections
import itertools

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_distance(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    best_routes = None
    best_max = float('inf')

    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        new_max = max(route_distance(r) for r in routes)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [list(r) for r in routes]

    # Construction: farthest-first clustering + nearest neighbor routing
    if truck_count >= len(customers):
        clusters = [[c] for c in customers]
        while len(clusters) < truck_count:
            clusters.append([])
    else:
        seeds = []
        # first seed: farthest from depot
        first_seed = max(customers, key=lambda x: distance_matrix[0][x])
        seeds.append(first_seed)
        while len(seeds) < truck_count:
            best_cust = None
            best_min_dist = -1
            for c in customers:
                if c in seeds:
                    continue
                min_dist = min(distance_matrix[c][s] for s in seeds)
                if min_dist > best_min_dist or (min_dist == best_min_dist and best_cust is not None and c < best_cust):
                    best_min_dist = min_dist
                    best_cust = c
            seeds.append(best_cust)
        # assign customers to nearest seed (tie-break smaller seed index)
        clusters = [[] for _ in range(truck_count)]
        for c in customers:
            if c in seeds:
                continue
            min_dist = float('inf')
            best_idx = 0
            for i, s in enumerate(seeds):
                d = distance_matrix[c][s]
                if d < min_dist or (d == min_dist and i < best_idx):
                    min_dist = d
                    best_idx = i
            clusters[best_idx].append(c)
        for i, s in enumerate(seeds):
            clusters[i].append(s)

    # build initial routes using nearest neighbor from depot
    routes = []
    for cl in clusters:
        if not cl:
            routes.append([0, 0])
        else:
            route = [0]
            unvisited = list(cl)
            current = 0
            while unvisited:
                nearest = min(unvisited, key=lambda x: distance_matrix[current][x])
                route.append(nearest)
                unvisited.remove(nearest)
                current = nearest
            route.append(0)
            routes.append(route)
    report_best_vrp(routes)

    # Tabu search parameters
    max_iter = min(500, n * truck_count)
    tabu_tenure = max(5, n // 10)
    tabu_list = collections.deque(maxlen=tabu_tenure)
    no_improve = 0

    for iteration in range(max_iter):
        improved = False
        dists = [route_distance(r) for r in routes]
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
        interior = routes[max_idx][1:-1]
        if not interior:
            break

        # Evaluate all non-tabu moves: relocate and swap
        best_move = None
        best_new_max = float('inf')

        # Relocate moves from max route to other routes
        for cust in interior:
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                move_key = (cust, max_idx, other_idx)
                if move_key in tabu_list:
                    continue
                other_route = routes[other_idx]
                for pos in range(1, len(other_route)):
                    new_routes = [list(r) for r in routes]
                    new_routes[max_idx].remove(cust)
                    new_routes[other_idx].insert(pos, cust)
                    new_max = max(route_distance(r) for r in new_routes)
                    if new_max < best_new_max - 1e-12:
                        best_new_max = new_max
                        best_move = ('relocate', cust, max_idx, other_idx, pos, new_routes)
                    elif abs(new_max - best_new_max) < 1e-12:
                        # tie-break: smaller cust, then other_idx, then pos
                        if (cust < best_move[1] or 
                            (cust == best_move[1] and other_idx < best_move[3]) or
                            (cust == best_move[1] and other_idx == best_move[3] and pos < best_move[4])):
                            best_new_max = new_max
                            best_move = ('relocate', cust, max_idx, other_idx, pos, new_routes)

        # Swap moves between max route and other routes
        for other_idx in range(truck_count):
            if other_idx == max_idx:
                continue
            other_interior = routes[other_idx][1:-1]
            if not other_interior:
                continue
            for cust_max in interior:
                for cust_other in other_interior:
                    move_key = (cust_max, cust_other)
                    reversed_key = (cust_other, cust_max)
                    if move_key in tabu_list or reversed_key in tabu_list:
                        continue
                    new_routes = [list(r) for r in routes]
                    idx_max = new_routes[max_idx].index(cust_max)
                    idx_other = new_routes[other_idx].index(cust_other)
                    new_routes[max_idx][idx_max] = cust_other
                    new_routes[other_idx][idx_other] = cust_max
                    new_max = max(route_distance(r) for r in new_routes)
                    if new_max < best_new_max - 1e-12:
                        best_new_max = new_max
                        best_move = ('swap', cust_max, max_idx, cust_other, other_idx, new_routes)
                    elif abs(new_max - best_new_max) < 1e-12:
                        # tie-break: smaller cust_max, then cust_other
                        if (cust_max < best_move[1] or 
                            (cust_max == best_move[1] and cust_other < best_move[3])):
                            best_new_max = new_max
                            best_move = ('swap', cust_max, max_idx, cust_other, other_idx, new_routes)

        # Apply best move
        if best_move is not None:
            # Aspiration: if new_max improves global best, accept even if tabu
            if best_new_max < best_max - 1e-12:
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
                # non-tabu move (already ensured)
                if best_move[0] == 'relocate':
                    _, cust, from_idx, to_idx, pos, new_routes = best_move
                    tabu_list.append((cust, from_idx, to_idx))
                else:
                    _, cust_max, max_idx, cust_other, other_idx, new_routes = best_move
                    tabu_list.append((cust_max, cust_other))
                routes = new_routes
                improved = True

        # Intra-route 2-opt after move
        for idx in range(truck_count):
            route = routes[idx]
            if len(route) <= 3:
                continue
            best_route = route[:]
            best_dist = route_distance(route)
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
            if found:
                routes[idx] = best_route
                new_max = max(route_distance(r) for r in routes)
                if new_max < best_max - 1e-12:
                    report_best_vrp(routes)
                improved = True

        if improved:
            no_improve = 0
        else:
            no_improve += 1

        # If no improvement for a while, trigger ruin-recreate
        if no_improve >= 3:
            no_improve = 0
            # Find longest route
            dists = [route_distance(r) for r in routes]
            max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
            interior = routes[max_idx][1:-1]
            if len(interior) < 1:
                continue
            remove_cnt = min(1 + iteration // 10, len(interior))
            remove_cnt = min(remove_cnt, 4)
            # Remove first few customers (by index) for determinism
            to_remove = sorted(interior)[:remove_cnt]
            # Remove from routes
            new_routes = []
            for r in routes:
                new_route = [c for c in r if c not in to_remove]
                if new_route[0] != 0:
                    new_route = [0] + new_route
                if new_route[-1] != 0:
                    new_route.append(0)
                new_routes.append(new_route)
            # Repair using regret-2 insertion
            unassigned = sorted(to_remove)
            while unassigned:
                best_regret = -1e100
                best_cust = None
                best_route_idx = None
                best_pos = None
                for cust in unassigned:
                    insertions = []
                    for r_idx, route in enumerate(new_routes):
                        interior_list = route[1:-1]
                        if not interior_list:
                            # empty route
                            delta = distance_matrix[0][cust] + distance_matrix[cust][0]
                            insertions.append((delta, r_idx, 0))
                        else:
                            best_delta = float('inf')
                            best_p = 0
                            for pos in range(len(interior_list)+1):
                                if pos == 0:
                                    prev = 0
                                    nxt = interior_list[0]
                                elif pos == len(interior_list):
                                    prev = interior_list[-1]
                                    nxt = 0
                                else:
                                    prev = interior_list[pos-1]
                                    nxt = interior_list[pos]
                                delta = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                                if delta < best_delta:
                                    best_delta = delta
                                    best_p = pos
                            insertions.append((best_delta, r_idx, best_p))
                    insertions.sort(key=lambda x: (x[0], x[2]))
                    best = insertions[0][0]
                    second = insertions[1][0] if len(insertions) > 1 else best
                    regret = second - best
                    if regret > best_regret or (regret == best_regret and (best_cust is None or cust < best_cust)):
                        best_regret = regret
                        best_cust = cust
                        best_route_idx = insertions[0][1]
                        best_pos = insertions[0][2]
                if best_cust is None:
                    break
                # Insert into route
                route = new_routes[best_route_idx]
                interior_list = route[1:-1]
                new_interior = interior_list[:best_pos] + [best_cust] + interior_list[best_pos:]
                new_routes[best_route_idx] = [0] + new_interior + [0]
                unassigned.remove(best_cust)
            new_max = max(route_distance(r) for r in new_routes)
            if new_max < best_max - 1e-12:
                routes = new_routes
                report_best_vrp(routes)
                improved = True

        if not improved and no_improve >= 5:
            break

    final_routes = best_routes if best_routes is not None else routes
    while len(final_routes) < truck_count:
        final_routes.append([0, 0])
    return final_routes