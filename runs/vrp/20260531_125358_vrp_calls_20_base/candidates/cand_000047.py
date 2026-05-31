def solve_vrp(distance_matrix, truck_count):
    import random
    import numpy as np
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    def report_best_vrp(routes):
        pass
    
    if truck_count >= n:
        routes = []
        for i in range(truck_count):
            if i < len(customers):
                routes.append([0, customers[i], 0])
            else:
                routes.append([0,0])
        return routes
    
    def route_length(route):
        total = 0
        for i in range(len(route)-1):
            total += distance_matrix[route[i]][route[i+1]]
        return total
    
    def two_opt(route, max_iter=10):
        route = route[:]
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            best_improvement = 0
            best_route = route
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    improvement = route_length(route) - route_length(new_route)
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_route = new_route
                        improved = True
            route = best_route
        return route
    
    def balance(routes, lengths):
        improved = True
        bal_iter = 0
        max_bal_iter = n * truck_count
        no_improve = 0
        while improved and bal_iter < max_bal_iter and no_improve < 2:
            improved = False
            bal_iter += 1
            max_idx = max(range(truck_count), key=lambda i: lengths[i])
            min_idx = min(range(truck_count), key=lambda i: lengths[i])
            if max_idx == min_idx or lengths[max_idx] == lengths[min_idx]:
                break
            max_route = routes[max_idx]
            min_route = routes[min_idx]
            best_move = None
            best_reduction = 0
            for pos in range(1, len(max_route)-1):
                cust = max_route[pos]
                new_max = [node for node in max_route if node != cust]
                new_max_len = route_length(new_max)
                for ins in range(1, len(min_route)):
                    new_min = min_route[:ins] + [cust] + min_route[ins:]
                    new_min_len = route_length(new_min)
                    other_lengths = [lengths[i] for i in range(truck_count) if i not in (max_idx, min_idx)]
                    new_max_global = max(new_max_len, new_min_len, max(other_lengths) if other_lengths else 0)
                    reduction = max(lengths) - new_max_global
                    if reduction > best_reduction:
                        best_reduction = reduction
                        best_move = (cust, pos, ins)
            if best_move:
                cust, pos, ins = best_move
                routes[max_idx] = [node for node in max_route if node != cust]
                routes[min_idx] = min_route[:ins] + [cust] + min_route[ins:]
                lengths[max_idx] = route_length(routes[max_idx])
                lengths[min_idx] = route_length(routes[min_idx])
                improved = True
                no_improve = 0
                report_best_vrp(routes)
            else:
                no_improve += 1
        return routes, lengths
    
    best_routes = None
    best_max = float('inf')
    num_restarts = min(20, max(1, n//5))
    for restart in range(num_restarts):
        random.seed(restart)
        # nearest neighbor clustering with deterministic tie-breaking
        seeds = random.sample(customers, min(truck_count, len(customers)))
        clusters = [[] for _ in range(truck_count)]
        for i, s in enumerate(seeds):
            clusters[i].append(s)
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
        # build routes with nearest neighbor within cluster
        routes = []
        for i in range(truck_count):
            if not clusters[i]:
                routes.append([0,0])
            else:
                unvisited = set(clusters[i])
                route = [0]
                current = 0
                while unvisited:
                    next_node = min(unvisited, key=lambda x: distance_matrix[current][x])
                    route.append(next_node)
                    unvisited.remove(next_node)
                    current = next_node
                route.append(0)
                routes.append(route)
        # local search
        for i in range(truck_count):
            if len(routes[i]) > 2:
                routes[i] = two_opt(routes[i], max_iter=len(clusters[i])*2)
        lengths = [route_length(r) for r in routes]
        routes, lengths = balance(routes, lengths)
        cur_max = max(lengths)
        if cur_max < best_max:
            best_max = cur_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
    if best_routes is None:
        best_routes = [[0,0] for _ in range(truck_count)]
        idx = 0
        for c in customers:
            if idx < truck_count:
                best_routes[idx] = [0, c, 0]
                idx += 1
            else:
                best_routes[-1].insert(-1, c)
    return best_routes