import numpy as np
import random
import time

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes

    def route_distance(route, dm):
        if len(route) <= 1:
            return 0
        return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))

    def greedy_insert(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        for cust in perm:
            best_truck = None
            best_max_val = float('inf')
            best_insert_pos = None
            for t in range(truck_count):
                route = routes[t]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_dist = route_distance(new_route, distance_matrix)
                    max_other = 0
                    for tt in range(truck_count):
                        if tt == t:
                            continue
                        max_other = max(max_other, route_distance(routes[tt], distance_matrix))
                    new_max = max(max_other, new_dist)
                    if new_max < best_max_val:
                        best_max_val = new_max
                        best_truck = t
                        best_insert_pos = pos
            if best_truck is not None:
                route = routes[best_truck]
                routes[best_truck] = route[:best_insert_pos] + [cust] + route[best_insert_pos:]
            else:
                routes[0].insert(-1, cust)
        return routes

    def intra_2opt(route, dm, max_iter=50):
        if len(route) <= 3:
            return route
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route, dm)
                    old_dist = route_distance(route, dm)
                    if new_dist < old_dist - 1e-10:
                        route = new_route
                        improved = True
            it += 1
        return route

    best_routes = None
    best_max = float('inf')
    start_time = time.time()
    time_limit = 175
    restarts = max(3, min(10, n//10))

    for restart in range(restarts):
        if time.time() - start_time > time_limit:
            break
        perm = customers[:]
        random.shuffle(perm)
        routes = greedy_insert(perm)

        for t in range(truck_count):
            routes[t] = intra_2opt(routes[t], distance_matrix)

        # Inter-route improvement
        improved = True
        inter_it = 0
        while improved and inter_it < 50 and (time.time() - start_time) < time_limit:
            improved = False
            current_max = max(route_distance(r, distance_matrix) for r in routes)
            best_move = None
            best_new_max = current_max
            for t1 in range(truck_count):
                for t2 in range(truck_count):
                    if t1 == t2:
                        continue
                    route1 = routes[t1]
                    route2 = routes[t2]
                    if len(route1) <= 2 or len(route2) <= 2:
                        continue
                    # Relocate
                    for i in range(1, len(route1)-1):
                        cust = route1[i]
                        for j in range(1, len(route2)):
                            new_route1 = route1[:i] + route1[i+1:]
                            new_route2 = route2[:j] + [cust] + route2[j:]
                            new_routes = routes[:]
                            new_routes[t1] = new_route1
                            new_routes[t2] = new_route2
                            new_max = max(route_distance(r, distance_matrix) for r in new_routes)
                            if new_max < best_new_max:
                                best_new_max = new_max
                                best_move = ('relocate', t1, i, t2, j)
                    # Swap
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new_route1 = route1[:i] + [route2[j]] + route1[i+1:]
                            new_route2 = route2[:j] + [route1[i]] + route2[j+1:]
                            new_routes = routes[:]
                            new_routes[t1] = new_route1
                            new_routes[t2] = new_route2
                            new_max = max(route_distance(r, distance_matrix) for r in new_routes)
                            if new_max < best_new_max:
                                best_new_max = new_max
                                best_move = ('swap', t1, i, t2, j)
                    # 2-opt*
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new_route1 = route1[:i+1] + route2[j+1:-1] + [0]
                            new_route2 = route2[:j+1] + route1[i+1:-1] + [0]
                            new_routes = routes[:]
                            new_routes[t1] = new_route1
                            new_routes[t2] = new_route2
                            new_max = max(route_distance(r, distance_matrix) for r in new_routes)
                            if new_max < best_new_max:
                                best_new_max = new_max
                                best_move = ('2opt*', t1, i, t2, j)
            if best_move is not None:
                kind, t1, i, t2, j = best_move
                if kind == 'relocate':
                    cust = routes[t1][i]
                    routes[t1].pop(i)
                    routes[t2].insert(j, cust)
                elif kind == 'swap':
                    cust1 = routes[t1][i]
                    cust2 = routes[t2][j]
                    routes[t1][i] = cust2
                    routes[t2][j] = cust1
                elif kind == '2opt*':
                    orig1 = routes[t1]
                    orig2 = routes[t2]
                    routes[t1] = orig1[:i+1] + orig2[j+1:-1] + [0]
                    routes[t2] = orig2[:j+1] + orig1[i+1:-1] + [0]
                improved = True
                inter_it += 1
                if best_new_max < best_max - 1e-10:
                    best_max = best_new_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)

        # After improvement, check if better
        current_max = max(route_distance(r, distance_matrix) for r in routes)
        if current_max < best_max - 1e-10:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

    if best_routes is None:
        routes = [[0, 0] for _ in range(truck_count)]
        for idx, cust in enumerate(customers):
            routes[idx % truck_count].insert(-1, cust)
        best_routes = routes
        report_best_vrp(best_routes)
    return best_routes