import numpy as np

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

    # Construction: each customer as a separate route
    route_list = [[0, c, 0] for c in customers]

    # Merge until we have exactly truck_count routes
    while len(route_list) > truck_count:
        n_routes = len(route_list)
        merges = []  # each: (-saving, new_max, ri, rj, new_route)
        for ri in range(n_routes):
            Ri = route_list[ri]
            interior_i = Ri[1:-1]
            if not interior_i:
                continue
            first_i = interior_i[0]
            last_i = interior_i[-1]
            for rj in range(ri+1, n_routes):
                Rj = route_list[rj]
                interior_j = Rj[1:-1]
                if not interior_j:
                    continue
                first_j = interior_j[0]
                last_j = interior_j[-1]
                if last_i == first_j:
                    # forward merge
                    new_route = Ri[:-1] + Rj[1:]
                    saving = distance_matrix[0, last_i] + distance_matrix[0, first_j] - distance_matrix[last_i, first_j]
                elif last_j == first_i:
                    # reverse merge (swap roles)
                    new_route = Rj[:-1] + Ri[1:]
                    saving = distance_matrix[0, last_j] + distance_matrix[0, first_i] - distance_matrix[last_j, first_i]
                else:
                    continue
                # compute new max distance after merge
                new_routes = [route_list[k] for k in range(n_routes) if k != ri and k != rj]
                new_routes.append(new_route)
                new_max = max(route_distance(r) for r in new_routes)
                merges.append((-saving, new_max, ri, rj, new_route))
        if not merges:
            break
        # sort by saving (most negative first) then new max
        merges.sort(key=lambda x: (x[0], x[1]))
        _, _, ri, rj, new_route = merges[0]
        # rebuild route list with merged route
        new_route_list = [route_list[k] for k in range(len(route_list)) if k != ri and k != rj]
        new_route_list.append(new_route)
        route_list = new_route_list

    report_best_vrp(route_list)

    # Adaptive improvement schedule
    max_iter = max(200, len(customers) * truck_count)
    move_sequence = ['relocate', 'swap', '2opt'] * (max_iter // 3 + 1)
    for move_type in move_sequence[:max_iter]:
        improved = False
        dists = [route_distance(r) for r in route_list]
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
        max_route = route_list[max_idx]
        interior = max_route[1:-1]
        if not interior:
            break

        if move_type == 'relocate':
            for cust in interior:
                for other_idx in range(truck_count):
                    if other_idx == max_idx:
                        continue
                    other_route = route_list[other_idx]
                    best_pos = 0
                    best_delta = float('inf')
                    for pos in range(1, len(other_route)):
                        prev = other_route[pos-1]
                        nxt = other_route[pos]
                        delta = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        if delta < best_delta:
                            best_delta = delta
                            best_pos = pos
                    if best_delta < 0:
                        new_routes = [list(r) for r in route_list]
                        new_routes[max_idx].remove(cust)
                        new_routes[other_idx].insert(best_pos, cust)
                        new_max = max(route_distance(r) for r in new_routes)
                        if new_max < best_max - 1e-12:
                            route_list = new_routes
                            report_best_vrp(route_list)
                            improved = True
                            break
                if improved:
                    break
        elif move_type == 'swap':
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = route_list[other_idx]
                interior_other = other_route[1:-1]
                if not interior_other:
                    continue
                for cust_max in interior:
                    for cust_other in interior_other:
                        new_routes = [list(r) for r in route_list]
                        idx_max = new_routes[max_idx].index(cust_max)
                        idx_other = new_routes[other_idx].index(cust_other)
                        new_routes[max_idx][idx_max] = cust_other
                        new_routes[other_idx][idx_other] = cust_max
                        new_max = max(route_distance(r) for r in new_routes)
                        if new_max < best_max - 1e-12:
                            route_list = new_routes
                            report_best_vrp(route_list)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        elif move_type == '2opt':
            for idx in range(truck_count):
                route = route_list[idx]
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
                    route_list[idx] = best_route
                    new_max = max(route_distance(r) for r in route_list)
                    if new_max < best_max - 1e-12:
                        report_best_vrp(route_list)
                    improved = True
                    break
        if not improved:
            continue

    final_routes = best_routes if best_routes is not None else route_list
    while len(final_routes) < truck_count:
        final_routes.append([0, 0])
    return final_routes