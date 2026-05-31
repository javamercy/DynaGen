import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n:
        routes = []
        for i in range(truck_count):
            if i < len(customers):
                routes.append([0, customers[i], 0])
            else:
                routes.append([0, 0])
        return routes

    # farthest-first seed selection based on depot distance, then evenly spaced
    sorted_customers = sorted(customers, key=lambda c: distance_matrix[0][c], reverse=True)
    clusters = [[] for _ in range(truck_count)]
    seeds = []
    step = len(sorted_customers) // truck_count
    for i in range(truck_count):
        idx = i * step if i * step < len(sorted_customers) else len(sorted_customers) - 1
        seeds.append(sorted_customers[idx])
        clusters[i].append(sorted_customers[idx])
    remaining = [c for c in customers if c not in seeds]
    for cust in remaining:
        best_cluster = 0
        best_dist = distance_matrix[cust][seeds[0]]
        for i in range(1, truck_count):
            d = distance_matrix[cust][seeds[i]]
            if d < best_dist or (d == best_dist and i < best_cluster):
                best_dist = d
                best_cluster = i
        clusters[best_cluster].append(cust)

    def route_length(route):
        total = 0
        for i in range(len(route)-1):
            total += distance_matrix[route[i]][route[i+1]]
        return total

    def build_route(cluster):
        if not cluster:
            return [0, 0]
        route = [0]
        unvisited = set(cluster)
        current = 0
        while unvisited:
            next_node = min(unvisited, key=lambda x: distance_matrix[current][x])
            route.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        route.append(0)
        return route

    def two_opt(route, max_iter=10):
        route = route[:]
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_length(new_route) < route_length(route):
                        route = new_route
                        improved = True
        return route

    routes = []
    for i in range(truck_count):
        if not clusters[i]:
            routes.append([0, 0])
        else:
            route = build_route(clusters[i])
            route = two_opt(route, max_iter=10)
            routes.append(route)

    lengths = [route_length(r) for r in routes]
    # try one balancing move: move a customer from longest to shortest route if it reduces max
    max_idx = max(range(truck_count), key=lambda i: lengths[i])
    min_idx = min(range(truck_count), key=lambda i: lengths[i])
    if max_idx != min_idx and lengths[max_idx] > lengths[min_idx]:
        max_route = routes[max_idx]
        best_cust = None
        best_pos_remove = None
        best_pos_insert = None
        best_reduction = 0
        for pos in range(1, len(max_route)-1):
            cust = max_route[pos]
            new_max_route = max_route[:pos] + max_route[pos+1:]
            new_max_len = route_length(new_max_route)
            min_route = routes[min_idx]
            for p in range(1, len(min_route)):
                new_min_route = min_route[:p] + [cust] + min_route[p:]
                new_min_len = route_length(new_min_route)
                other_lengths = [lengths[i] for i in range(truck_count) if i not in (max_idx, min_idx)]
                new_max_global = max(new_max_len, new_min_len, max(other_lengths) if other_lengths else 0)
                old_max_global = max(lengths)
                reduction = old_max_global - new_max_global
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_cust = cust
                    best_pos_remove = pos
                    best_pos_insert = p
        if best_cust is not None:
            new_max = [node for node in max_route if node != best_cust]
            min_route = routes[min_idx]
            new_min = min_route[:best_pos_insert] + [best_cust] + min_route[best_pos_insert:]
            routes[max_idx] = new_max
            routes[min_idx] = new_min
            lengths[max_idx] = route_length(new_max)
            lengths[min_idx] = route_length(new_min)
            report_best_vrp(routes)

    # final 2-opt on all routes
    for i in range(truck_count):
        if len(routes[i]) > 2:
            routes[i] = two_opt(routes[i], max_iter=10)

    return routes