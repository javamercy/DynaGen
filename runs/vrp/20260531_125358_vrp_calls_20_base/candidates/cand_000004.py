def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n:
        routes = []
        for i in range(truck_count):
            if i < len(customers):
                routes.append([0, customers[i], 0])
            else:
                routes.append([0,0])
        return routes
    sorted_customers = sorted(customers, key=lambda c: distance_matrix[0][c], reverse=True)
    clusters = [[] for _ in range(truck_count)]
    seeds = []
    for i in range(truck_count):
        clusters[i].append(sorted_customers[i])
        seeds.append(sorted_customers[i])
    remaining = [c for c in customers if c not in seeds]
    for cust in remaining:
        best_cluster = 0
        best_dist = distance_matrix[cust][seeds[0]]
        for i in range(1, truck_count):
            d = distance_matrix[cust][seeds[i]]
            if d < best_dist:
                best_dist = d
                best_cluster = i
            elif d == best_dist and i < best_cluster:
                best_cluster = i
        clusters[best_cluster].append(cust)
    def route_length(route):
        total = 0
        for i in range(len(route)-1):
            total += distance_matrix[route[i]][route[i+1]]
        return total
    def build_route(cluster):
        if not cluster:
            return [0,0]
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
            routes.append([0,0])
        else:
            route = build_route(clusters[i])
            route = two_opt(route, max_iter=len(clusters[i])*2)
            routes.append(route)
    lengths = [route_length(r) for r in routes]
    for _ in range(len(customers) * truck_count):
        max_idx = max(range(truck_count), key=lambda i: lengths[i])
        min_idx = min(range(truck_count), key=lambda i: lengths[i])
        if max_idx == min_idx or lengths[max_idx] == lengths[min_idx]:
            break
        max_route = routes[max_idx]
        best_cust = None
        best_overall_reduction = 0
        for pos in range(1, len(max_route)-1):
            cust = max_route[pos]
            new_max_route = max_route[:pos] + max_route[pos+1:]
            new_max_len = route_length(new_max_route)
            min_route = routes[min_idx]
            best_insertion_len = float('inf')
            best_pos = -1
            for p in range(1, len(min_route)):
                new_min_route = min_route[:p] + [cust] + min_route[p:]
                l = route_length(new_min_route)
                if l < best_insertion_len:
                    best_insertion_len = l
                    best_pos = p
            new_min_route = min_route[:best_pos] + [cust] + min_route[best_pos:]
            new_min_len = route_length(new_min_route)
            other_lengths = [lengths[i] for i in range(truck_count) if i not in (max_idx, min_idx)]
            new_max_global = max(new_max_len, new_min_len, max(other_lengths) if other_lengths else 0)
            old_max_global = max(lengths)
            reduction = old_max_global - new_max_global
            if reduction > best_overall_reduction:
                best_overall_reduction = reduction
                best_cust = (cust, pos, best_pos)
        if best_cust is not None:
            cust, pos_remove, pos_insert = best_cust
            new_max = [node for node in max_route if node != cust]
            min_route = routes[min_idx]
            new_min = min_route[:pos_insert] + [cust] + min_route[pos_insert:]
            routes[max_idx] = new_max
            routes[min_idx] = new_min
            lengths[max_idx] = route_length(new_max)
            lengths[min_idx] = route_length(new_min)
            report_best_vrp(routes)
        else:
            break
    return routes