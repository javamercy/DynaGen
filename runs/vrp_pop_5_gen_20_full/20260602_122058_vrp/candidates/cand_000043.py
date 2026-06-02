import numpy as np

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

    def max_dist():
        return max(route_dist(r) for r in routes)

    # Step 1: farthest-first seed selection
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

    # Step 2: initialize routes with seeds
    routes = [[0, s, 0] for s in seeds]
    remaining = [c for c in customers if c not in seeds]
    remaining.sort(key=lambda c: -distance_matrix[0, c])

    # Step 3: greedily insert remaining customers minimizing max distance
    for cust in remaining:
        best_new_max = float('inf')
        best_route_idx = -1
        best_pos = -1
        for idx, route in enumerate(routes):
            best_delta = float('inf')
            best_pos_local = -1
            for pos in range(1, len(route)):
                prev = route[pos-1]
                nex = route[pos]
                delta = distance_matrix[prev, cust] + distance_matrix[cust, nex] - distance_matrix[prev, nex]
                if delta < best_delta:
                    best_delta = delta
                    best_pos_local = pos
            if best_pos_local == -1:
                continue
            current_route_dist = route_dist(route)
            new_route_dist = current_route_dist + best_delta
            other_max = 0.0
            for j, r in enumerate(routes):
                if j == idx:
                    continue
                other_max = max(other_max, route_dist(r))
            new_max = max(other_max, new_route_dist)
            if new_max < best_new_max or (new_max == best_new_max and idx < best_route_idx):
                best_new_max = new_max
                best_route_idx = idx
                best_pos = best_pos_local
        routes[best_route_idx].insert(best_pos, cust)
        report_best_vrp(routes)

    report_best_vrp(routes)

    # Local search function
    def improve(routes):
        max_iter = min(50, n * truck_count)
        for _ in range(max_iter):
            improved = False
            current_max = max_dist()
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
                        if new_max < best_new_max or (new_max == best_new_max and (best_move is None or (i, pos, j, best_pos_j) < best_move[1:])):
                            best_new_max = new_max
                            best_move = ('relocate', i, pos, j, best_pos_j)
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
            best_new_max = max_dist()
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
                            if new_max < best_new_max or (new_max == best_new_max and (best_move is None or (i, pos_i, j, pos_j) < best_move[1:])):
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
            best_new_max = max_dist()
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
                        if new_max < best_new_max or (new_max == best_new_max and (best_move is None or (i, a, b) < best_move[1:])):
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
            best_new_max = max_dist()
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
                            if new_max < best_new_max or (new_max == best_new_max and (best_move is None or (i, j, cut_i, cut_j) < best_move[1:])):
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

    # First local search
    improve(routes)
    best_max = max_dist()
    best_routes = [list(r) for r in routes]

    # Perturbation function (strengthened: remove up to 3 customers with highest savings from longest route and reinsert greedily)
    def perturb(routes):
        max_route_idx = max(range(truck_count), key=lambda i: route_dist(routes[i]))
        route = routes[max_route_idx]
        if len(route) <= 3:
            return  # not enough customers to remove meaningfully
        # Compute savings for each customer in the route (excluding depot)
        removals = []
        for pos in range(1, len(route)-1):
            prev = route[pos-1]
            cust = route[pos]
            nex = route[pos+1]
            saving = distance_matrix[prev, cust] + distance_matrix[cust, nex] - distance_matrix[prev, nex]
            removals.append((saving, pos, cust))
        # Sort by saving descending, tie by cust descending for determinism
        removals.sort(key=lambda x: (-x[0], -x[2]))
        # Choose up to 3 removals (but not exceeding half of route length)
        max_remove = min(3, (len(route) - 2) // 2)
        selected = removals[:max_remove]
        # Remove them in reverse order of position to preserve indices
        selected.sort(key=lambda x: -x[1])
        removed_customers = []
        for saving, pos, cust in selected:
            route.pop(pos)
            removed_customers.append(cust)
        # Reinsert each removed customer greedily minimizing max distance
        for cust in removed_customers:
            best_new_max = float('inf')
            best_route_idx = -1
            best_ins_pos = -1
            for idx, r in enumerate(routes):
                best_delta = float('inf')
                best_pos_local = -1
                for ins_pos in range(1, len(r)):
                    prev = r[ins_pos-1]
                    nex = r[ins_pos]
                    delta = distance_matrix[prev, cust] + distance_matrix[cust, nex] - distance_matrix[prev, nex]
                    if delta < best_delta:
                        best_delta = delta
                        best_pos_local = ins_pos
                if best_pos_local == -1:
                    continue
                new_route_dist = route_dist(r) + best_delta
                other_max = 0.0
                for k, rr in enumerate(routes):
                    if k == idx:
                        continue
                    other_max = max(other_max, route_dist(rr))
                new_max = max(other_max, new_route_dist)
                if new_max < best_new_max or (new_max == best_new_max and idx < best_route_idx):
                    best_new_max = new_max
                    best_route_idx = idx
                    best_ins_pos = best_pos_local
            if best_route_idx == -1:
                # fallback: insert back into original route
                routes[max_route_idx].insert(1, cust)  # after depot
            else:
                routes[best_route_idx].insert(best_ins_pos, cust)

    # Restart loop
    restarts = 5
    for _ in range(restarts):
        new_routes = [list(r) for r in best_routes]
        perturb(new_routes)
        improve(new_routes)
        cur_max = max(route_dist(r) for r in new_routes)
        if cur_max < best_max:
            best_max = cur_max
            best_routes = [list(r) for r in new_routes]
            # report_best_vrp will be called inside improve already
            # but to be safe, call once:
            report_best_vrp(new_routes)

    return best_routes