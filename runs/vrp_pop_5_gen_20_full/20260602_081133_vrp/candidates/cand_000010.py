import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    routes = [[0] for _ in range(truck_count)]
    unassigned = list(range(1, n))

    def route_dist(route):
        if len(route) <= 1:
            return 0.0
        d = 0.0
        for i in range(len(route) - 1):
            d += distance_matrix[route[i]][route[i+1]]
        d += distance_matrix[route[-1]][0]
        return d

    # Construction
    while unassigned:
        best_truck = 0
        min_dist = route_dist(routes[0])
        for t in range(1, truck_count):
            d = route_dist(routes[t])
            if d < min_dist - 1e-12:
                min_dist = d
                best_truck = t
        last = routes[best_truck][-1]
        best_cust = None
        best_add = float('inf')
        for cust in unassigned:
            add = distance_matrix[last][cust] + distance_matrix[cust][0] - distance_matrix[last][0]
            if add < best_add - 1e-12:
                best_add = add
                best_cust = cust
        routes[best_truck].append(best_cust)
        unassigned.remove(best_cust)

    for t in range(truck_count):
        routes[t].append(0)

    distances = [route_dist(r) for r in routes]
    try:
        report_best_vrp(routes)
    except:
        pass

    # Improvement
    max_iter = n * truck_count * 10
    for _ in range(max_iter):
        improved = False
        current_max = max(distances)
        max_idx = 0
        max_dist = distances[0]
        for t in range(1, truck_count):
            if distances[t] > max_dist + 1e-12:
                max_dist = distances[t]
                max_idx = t

        best_move = None
        best_improvement = 0.0
        long_route = routes[max_idx]
        for pos in range(1, len(long_route) - 1):
            cust = long_route[pos]
            new_long = long_route[:pos] + long_route[pos+1:]
            new_long_dist = route_dist(new_long)
            for t in range(truck_count):
                if t == max_idx:
                    continue
                other_route = routes[t]
                for k in range(1, len(other_route)):
                    new_other = other_route[:k] + [cust] + other_route[k:]
                    new_other_dist = route_dist(new_other)
                    other_dists = [distances[i] for i in range(truck_count) if i not in (max_idx, t)]
                    new_max = max(new_long_dist, new_other_dist, *other_dists)
                    improvement = current_max - new_max
                    if improvement > best_improvement + 1e-12:
                        best_improvement = improvement
                        best_move = (max_idx, t, new_long, new_other, new_long_dist, new_other_dist)

        if best_improvement > 1e-12:
            _, _, new_long, new_other, d1, d2 = best_move
            routes[max_idx] = new_long
            routes[t] = new_other
            distances[max_idx] = d1
            distances[t] = d2
            try:
                report_best_vrp(routes)
            except:
                pass

            # Intra-route 2-opt on two affected routes
            for idx in [max_idx, t]:
                route = routes[idx]
                max_passes = min(len(route), 10)
                for _ in range(max_passes):
                    improved_2opt = False
                    for i in range(1, len(route) - 2):
                        for j in range(i + 1, len(route) - 1):
                            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                            if route_dist(new_route) < route_dist(route) - 1e-12:
                                route = new_route
                                improved_2opt = True
                    if not improved_2opt:
                        break
                routes[idx] = route
                distances[idx] = route_dist(route)
            try:
                report_best_vrp(routes)
            except:
                pass
            improved = True

        if not improved:
            break

    return routes