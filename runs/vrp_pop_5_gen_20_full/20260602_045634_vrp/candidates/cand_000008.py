import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    # Greedy insertion (same as parent)
    def route_distance(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    routes = [[0,0] for _ in range(truck_count)]
    distances = [0.0]*truck_count
    customers = list(range(1,n))
    for cust in customers:
        best_increase = float('inf')
        best_route = -1
        best_pos = -1
        for r in range(truck_count):
            route = routes[r]
            for pos in range(1, len(route)):
                new_dist = distances[r] + distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                new_max = max(new_dist, max(distances[:r] + distances[r+1:]))
                increase = new_max - max(distances)
                if increase < best_increase or (increase == best_increase and r < best_route):
                    best_increase = increase
                    best_route = r
                    best_pos = pos
        route = routes[best_route]
        route.insert(best_pos, cust)
        distances[best_route] = route_distance(route)

    best_routes = [list(r) for r in routes]
    best_max = max(distances)
    report_best_vrp(best_routes)

    # VND improvement
    def max_after_change(r, new_route, new_dist):
        return max(new_dist, max(distances[:r] + distances[r+1:]))

    improved = True
    total_iter = 0
    max_total_iter = n * truck_count * 150  # bounded
    while improved and total_iter < max_total_iter:
        improved = False
        total_iter += 1

        # 2-opt intra-route
        for r in range(truck_count):
            route = routes[r]
            loc_improved = True
            iter_2opt = 0
            max_2opt = len(route) * 2
            while loc_improved and iter_2opt < max_2opt:
                loc_improved = False
                iter_2opt += 1
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        new_max = max_after_change(r, new_route, new_dist)
                        if new_max < max(distances):
                            routes[r] = new_route
                            distances[r] = new_dist
                            loc_improved = True
                            if new_max < best_max:
                                best_routes = [list(r) for r in routes]
                                best_max = new_max
                                report_best_vrp(best_routes)
                            break
                    if loc_improved:
                        break
                if loc_improved:
                    improved = True

        # relocate inter-route
        for src in range(truck_count):
            route_src = routes[src]
            loc_improved = False
            iter_reloc = 0
            max_reloc = n * truck_count
            while not loc_improved and iter_reloc < max_reloc:
                iter_reloc += 1
                for pos_src in range(1, len(route_src)-1):
                    cust = route_src[pos_src]
                    temp_src = route_src[:pos_src] + route_src[pos_src+1:]
                    new_dist_src = route_distance(temp_src)
                    for dst in range(truck_count):
                        if dst == src:
                            continue
                        route_dst = routes[dst]
                        for pos_dst in range(1, len(route_dst)):
                            new_route_dst = route_dst[:pos_dst] + [cust] + route_dst[pos_dst:]
                            new_dist_dst = route_distance(new_route_dst)
                            new_max = max(new_dist_src, new_dist_dst, max([distances[i] for i in range(truck_count) if i != src and i != dst]))
                            if new_max < max(distances):
                                routes[src] = temp_src
                                distances[src] = new_dist_src
                                routes[dst] = new_route_dst
                                distances[dst] = new_dist_dst
                                loc_improved = True
                                improved = True
                                if new_max < best_max:
                                    best_routes = [list(r) for r in routes]
                                    best_max = new_max
                                    report_best_vrp(best_routes)
                                break
                        if loc_improved:
                            break
                    if loc_improved:
                        break
                if loc_improved:
                    break

        # swap inter-route
        for r1 in range(truck_count):
            route1 = routes[r1]
            loc_improved = False
            iter_swap = 0
            max_swap = n * truck_count
            while not loc_improved and iter_swap < max_swap:
                iter_swap += 1
                for pos1 in range(1, len(route1)-1):
                    cust1 = route1[pos1]
                    for r2 in range(r1+1, truck_count):
                        route2 = routes[r2]
                        for pos2 in range(1, len(route2)-1):
                            cust2 = route2[pos2]
                            # build new routes
                            new_route1 = route1[:pos1] + [cust2] + route1[pos1+1:]
                            new_route2 = route2[:pos2] + [cust1] + route2[pos2+1:]
                            new_dist1 = route_distance(new_route1)
                            new_dist2 = route_distance(new_route2)
                            new_max = max(new_dist1, new_dist2, max([distances[i] for i in range(truck_count) if i != r1 and i != r2]))
                            if new_max < max(distances):
                                routes[r1] = new_route1
                                distances[r1] = new_dist1
                                routes[r2] = new_route2
                                distances[r2] = new_dist2
                                loc_improved = True
                                improved = True
                                if new_max < best_max:
                                    best_routes = [list(r) for r in routes]
                                    best_max = new_max
                                    report_best_vrp(best_routes)
                                break
                        if loc_improved:
                            break
                    if loc_improved:
                        break
                if loc_improved:
                    break

    return best_routes