import random
import numpy as np

def solve_vrp(distance_matrix, truck_count):
    random.seed(0)
    n = distance_matrix.shape[0]
    if truck_count >= n - 1:
        routes = [[0, i, 0] for i in range(1, n)]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_dist(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_route_dist(routes):
        return max(route_dist(r) for r in routes)

    def greedy_insertion(routes, customer):
        best_inc = float('inf')
        best_ri = -1
        best_pos = -1
        for ri, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_dist = route_dist(route[:pos] + [customer] + route[pos:])
                old_dist = route_dist(route)
                inc = new_dist - old_dist
                if inc < best_inc or (inc == best_inc and (ri < best_ri or (ri == best_ri and pos < best_pos))):
                    best_inc = inc
                    best_ri = ri
                    best_pos = pos
        routes[best_ri].insert(best_pos, customer)
        return routes

    def random_construction():
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        for c in customers:
            routes = greedy_insertion(routes, c)
        return routes

    best_routes = None
    best_max = float('inf')
    num_restarts = 5
    for restart in range(num_restarts):
        routes = random_construction()
        improved = True
        max_iter = n * truck_count
        iteration = 0
        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            for cust in range(1, n):
                ri = None
                pos = None
                for ridx, route in enumerate(routes):
                    if cust in route:
                        ri = ridx
                        pos = route.index(cust)
                        break
                if ri is None:
                    continue
                old_route = routes[ri][:]
                routes[ri].pop(pos)
                best_ri = -1
                best_pos = -1
                best_new_max = float('inf')
                for other_ri, other_route in enumerate(routes):
                    if other_ri == ri:
                        continue
                    for p in range(1, len(other_route)):
                        new_other = other_route[:p] + [cust] + other_route[p:]
                        new_max = max(route_dist(routes[ri]) if len(routes[ri]) > 1 else 0,
                                      route_dist(new_other),
                                      *[route_dist(r) for idx, r in enumerate(routes) if idx not in (ri, other_ri)])
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_ri = other_ri
                            best_pos = p
                if best_new_max < max_route_dist(routes) and best_ri != -1:
                    routes[best_ri].insert(best_pos, cust)
                    improved = True
                    cur_max = max_route_dist(routes)
                    if cur_max < best_max:
                        best_max = cur_max
                        best_routes = [list(r) for r in routes]
                        report_best_vrp(best_routes)
                else:
                    routes[ri] = old_route
        cur_max = max_route_dist(routes)
        if cur_max < best_max:
            best_max = cur_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
    if best_routes is None:
        best_routes = random_construction()
    return best_routes