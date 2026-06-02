import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    num_customers = n - 1
    if truck_count < 1:
        return []
    
    customers = list(range(1, n))
    routes = [[0, c, 0] for c in customers]
    while len(routes) < truck_count:
        routes.append([0, 0])
    
    if num_customers == 0:
        while len(routes) < truck_count:
            routes.append([0,0])
        routes = routes[:truck_count]
        return routes
    
    savings = []
    for i in customers:
        for j in customers:
            if i < j:
                s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
                savings.append((s, i, j))
    savings.sort(key=lambda x: (-x[0], x[1], x[2]))
    
    cust_to_route = {}
    route_first = {}
    route_last = {}
    for idx, route in enumerate(routes):
        if len(route) > 2:
            first = route[1]
            last = route[-2]
            route_first[idx] = first
            route_last[idx] = last
            cust_to_route[first] = idx
            cust_to_route[last] = idx
        else:
            route_first[idx] = None
            route_last[idx] = None
    
    idx = 0
    while len(routes) > truck_count and idx < len(savings):
        s, i, j = savings[idx]
        idx += 1
        if i not in cust_to_route or j not in cust_to_route:
            continue
        ri = cust_to_route[i]
        rj = cust_to_route[j]
        if ri == rj:
            continue
        route_i = routes[ri]
        route_j = routes[rj]
        i_first = (route_i[1] == i)
        i_last = (route_i[-2] == i)
        j_first = (route_j[1] == j)
        j_last = (route_j[-2] == j)
        if not ((i_first or i_last) and (j_first or j_last)):
            continue
        if i_last and j_first:
            new_route = route_i[:-1] + route_j[1:]
        elif i_first and j_last:
            new_route = route_j[:-1] + route_i[1:]
        else:
            continue
        new_routes = []
        for idx_r, r in enumerate(routes):
            if idx_r != ri and idx_r != rj:
                new_routes.append(r)
        new_routes.append(new_route)
        routes = new_routes
        cust_to_route = {}
        route_first = {}
        route_last = {}
        for idx_r, r in enumerate(routes):
            if len(r) > 2:
                first = r[1]
                last = r[-2]
                route_first[idx_r] = first
                route_last[idx_r] = last
                cust_to_route[first] = idx_r
                cust_to_route[last] = idx_r
            else:
                route_first[idx_r] = None
                route_last[idx_r] = None
    
    while len(routes) < truck_count:
        routes.append([0,0])
    
    def route_dist(r):
        d = 0
        for a,b in zip(r, r[1:]):
            d += distance_matrix[a][b]
        return d
    
    report_best_vrp(routes)
    
    current_max = max(route_dist(r) for r in routes)
    improved = True
    max_iter = 50
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        max_idx = max(range(len(routes)), key=lambda idx: route_dist(routes[idx]))
        max_dist = route_dist(routes[max_idx])
        best_max = max_dist
        best_change = None
        for pos, cust in enumerate(routes[max_idx][1:-1]):
            new_max_route = routes[max_idx][:pos+1] + routes[max_idx][pos+2:]
            for other_idx in range(len(routes)):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                for ins_pos in range(1, len(other_route)):
                    new_other_route = other_route[:ins_pos] + [cust] + other_route[ins_pos:]
                    others = [route_dist(routes[i]) for i in range(len(routes)) if i not in (max_idx, other_idx)]
                    new_max_val = max(route_dist(new_max_route), route_dist(new_other_route), max(others) if others else 0)
                    if new_max_val < best_max:
                        best_max = new_max_val
                        best_change = ('move', max_idx, other_idx, pos, ins_pos, cust)
            for other_idx in range(len(routes)):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                for opos, ocust in enumerate(other_route[1:-1]):
                    new_max_swapped = routes[max_idx][:pos+1] + [ocust] + routes[max_idx][pos+2:]
                    new_other_swapped = other_route[:opos+1] + [cust] + other_route[opos+2:]
                    others = [route_dist(routes[i]) for i in range(len(routes)) if i not in (max_idx, other_idx)]
                    new_max_val = max(route_dist(new_max_swapped), route_dist(new_other_swapped), max(others) if others else 0)
                    if new_max_val < best_max:
                        best_max = new_max_val
                        best_change = ('swap', max_idx, other_idx, pos, opos, cust, ocust)
        if best_change is not None:
            if best_change[0] == 'move':
                _, max_idx, other_idx, pos, ins_pos, cust = best_change
                routes[max_idx] = routes[max_idx][:pos+1] + routes[max_idx][pos+2:]
                routes[other_idx] = routes[other_idx][:ins_pos] + [cust] + routes[other_idx][ins_pos:]
            else:
                _, max_idx, other_idx, pos, opos, cust, ocust = best_change
                routes[max_idx][pos+1] = ocust
                routes[other_idx][opos+1] = cust
            current_max = max(route_dist(r) for r in routes)
            improved = True
            report_best_vrp(routes)
    return routes