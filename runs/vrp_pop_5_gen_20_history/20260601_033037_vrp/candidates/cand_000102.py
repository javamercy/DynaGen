import numpy as np
import collections

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

    # Construction: greedy min-max insertion
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(customers)
    while unassigned:
        best_cust = None
        best_route_idx = None
        best_pos = None
        best_new_max = float('inf')
        sorted_cust = sorted(unassigned)
        for cust in sorted_cust:
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    temp_routes = [list(r) for r in routes]
                    temp_routes[r_idx] = new_route
                    new_max = max(route_distance(r) for r in temp_routes)
                    if new_max < best_new_max - 1e-12:
                        best_new_max = new_max
                        best_cust = cust
                        best_route_idx = r_idx
                        best_pos = pos
                    elif abs(new_max - best_new_max) < 1e-12:
                        if cust < best_cust or (cust == best_cust and r_idx < best_route_idx) or \
                           (cust == best_cust and r_idx == best_route_idx and pos < best_pos):
                            best_new_max = new_max
                            best_cust = cust
                            best_route_idx = r_idx
                            best_pos = pos
        routes[best_route_idx].insert(best_pos, best_cust)
        unassigned.remove(best_cust)
    # Apply 2-opt to each route after construction
    for idx in range(truck_count):
        route = routes[idx]
        if len(route) <= 3:
            continue
        improved = True
        while improved:
            improved = False
            for a in range(1, len(route)-2):
                for b in range(a+1, len(route)-1):
                    new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                    if route_distance(new_route) < route_distance(route) - 1e-12:
                        route = new_route
                        improved = True
                        break
                if improved:
                    break
        routes[idx] = route
    report_best_vrp(routes)

    # Tabu search parameters
    max_iter = min(500, n * truck_count)
    tabu_tenure = max(5, n // 10)
    tabu_list = collections.deque(maxlen=tabu_tenure)
    no_improve = 0

    for iteration in range(max_iter):
        improved = False
        # Find longest route
        dists = [route_distance(r) for r in routes]
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
        interior = routes[max_idx][1:-1]
        if not interior:
            break

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
                        if cust < best_move[1] if best_move else True:
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
                        if cust_max < best_move[1] if best_move else True:
                            best_new_max = new_max
                            best_move = ('swap', cust_max, max_idx, cust_other, other_idx, new_routes)

        if best_move is not None:
            if best_new_max < best_max - 1e-12:
                # Aspiration: accept even if tabu (but best_move is non-tabu by construction)
                if best_move[0] == 'relocate':
                    _, cust, from_idx, to_idx, pos, new_routes = best_move
                    tabu_list.append((cust, from_idx, to_idx))
                else:
                    _, cust_max, _, cust_other, _, new_routes = best_move
                    tabu_list.append((cust_max, cust_other))
                routes = new_routes
                report_best_vrp(routes)
                improved = True
            else:
                # Accept non-tabu move
                if best_move[0] == 'relocate':
                    _, cust, from_idx, to_idx, pos, new_routes = best_move
                    tabu_list.append((cust, from_idx, to_idx))
                else:
                    _, cust_max, _, cust_other, _, new_routes = best_move
                    tabu_list.append((cust_max, cust_other))
                routes = new_routes
                improved = True

        # Intra-route 2-opt on each route
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

        # Ruin-recreate perturbation
        if no_improve >= 3:
            no_improve = 0
            dists = [route_distance(r) for r in routes]
            max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
            interior = routes[max_idx][1:-1]
            if len(interior) < 1:
                continue
            remove_cnt = min(1 + iteration // 10, len(interior))
            remove_cnt = min(remove_cnt, 4)
            to_remove = sorted(interior)[:remove_cnt]
            # Remove customers from routes
            new_routes = []
            for r in routes:
                new_route = [c for c in r if c not in to_remove]
                if new_route[0] != 0:
                    new_route = [0] + new_route
                if new_route[-1] != 0:
                    new_route.append(0)
                new_routes.append(new_route)
            # Reinsert using cheapest insertion
            unassigned = sorted(to_remove)
            while unassigned:
                best_cust = None
                best_route_idx = None
                best_pos = None
                best_inc = float('inf')
                for cust in unassigned:
                    for r_idx, route in enumerate(new_routes):
                        interior_list = route[1:-1]
                        for pos in range(len(interior_list)+1):
                            if pos == 0:
                                prev = 0
                                nxt = interior_list[0] if interior_list else 0
                            elif pos == len(interior_list):
                                prev = interior_list[-1]
                                nxt = 0
                            else:
                                prev = interior_list[pos-1]
                                nxt = interior_list[pos]
                            inc = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                            if inc < best_inc - 1e-12 or (abs(inc - best_inc) < 1e-12 and (cust < best_cust or (cust == best_cust and r_idx < best_route_idx) or (cust == best_cust and r_idx == best_route_idx and pos < best_pos))):
                                best_inc = inc
                                best_cust = cust
                                best_route_idx = r_idx
                                best_pos = pos
                if best_cust is None:
                    break
                route = new_routes[best_route_idx]
                interior_list = route[1:-1]
                new_interior = interior_list[:best_pos] + [best_cust] + interior_list[best_pos:]
                new_routes[best_route_idx] = [0] + new_interior + [0]
                unassigned.remove(best_cust)
            new_max = max(route_distance(r) for r in new_routes)
            if new_max < best_max - 1e-12:
                routes = new_routes
                report_best_vrp(routes)

    return best_routes if best_routes is not None else routes