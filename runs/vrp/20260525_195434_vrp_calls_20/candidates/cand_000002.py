import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    customers.sort(key=lambda c: distance_matrix[0][c], reverse=True)
    routes = [[0, 0] for _ in range(truck_count)]
    
    def route_dist(route):
        return sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
    
    for cust in customers:
        best_increase = float('inf')
        best_route_idx = None
        best_pos = None
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                inc = distance_matrix[route[pos-1]][cust] + distance_matrix[cust][route[pos]] - distance_matrix[route[pos-1]][route[pos]]
                if inc < best_increase or (inc == best_increase and (r_idx < best_route_idx or (r_idx == best_route_idx and pos < best_pos))):
                    best_increase = inc
                    best_route_idx = r_idx
                    best_pos = pos
        routes[best_route_idx].insert(best_pos, cust)
    
    report_best_vrp(routes)
    
    # Intra-route 2-opt
    for idx in range(truck_count):
        route = routes[idx]
        if len(route) <= 2:
            continue
        improved = True
        max_iter = len(route) * len(route)
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    old = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                    new = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                    if new < old:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
                        iter_count += 1
                        break
                if improved:
                    break
        routes[idx] = route
    
    # Inter-route balancing
    def compute_max_dist():
        return max(route_dist(r) for r in routes)
    max_dist = compute_max_dist()
    improved = True
    max_outer_iter = n
    outer_iter = 0
    while improved and outer_iter < max_outer_iter:
        improved = False
        outer_iter += 1
        longest_idx = max(range(truck_count), key=lambda i: route_dist(routes[i]))
        longest_route = routes[longest_idx]
        if len(longest_route) <= 2:
            break
        for cust in longest_route[1:-1]:
            for other_idx in range(truck_count):
                if other_idx == longest_idx:
                    continue
                other_route = routes[other_idx]
                for pos in range(1, len(other_route)):
                    new_longest = longest_route.copy()
                    new_longest.remove(cust)
                    new_other = other_route.copy()
                    new_other.insert(pos, cust)
                    new_max = max(route_dist(new_longest), route_dist(new_other))
                    # compute max over all routes
                    for k in range(truck_count):
                        if k == longest_idx:
                            continue
                        if k == other_idx:
                            continue
                        d = route_dist(routes[k])
                        if d > new_max:
                            new_max = d
                    if new_max < max_dist:
                        routes[longest_idx] = new_longest
                        routes[other_idx] = new_other
                        max_dist = new_max
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    report_best_vrp(routes)
    return routes