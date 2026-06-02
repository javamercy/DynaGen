import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    
    # Initialize each customer as a separate route
    routes = [[0, i, 0] for i in range(1, n)]
    cust_to_route = {i: i-1 for i in range(1, n)}
    
    # Compute savings
    savings = []
    for i in range(1, n):
        for j in range(i+1, n):
            sav = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
            savings.append((sav, i, j))
    savings.sort(key=lambda x: (-x[0], x[1], x[2]))
    
    # Merge routes using savings
    for sav, i, j in savings:
        if len(routes) <= truck_count:
            break
        ri = cust_to_route[i]
        rj = cust_to_route[j]
        if ri == rj:
            continue
        # Check endpoints
        def first_last(route):
            return route[1], route[-2]
        fi, li = first_last(routes[ri])
        fj, lj = first_last(routes[rj])
        if (i == li and j == fj):
            new_route = routes[ri][:-1] + routes[rj][1:]
        elif (i == fi and j == lj):
            new_route = routes[rj][:-1] + routes[ri][1:]
        else:
            continue
        # Remove old routes (rj first if larger index)
        if ri > rj:
            ri, rj = rj, ri
        del routes[rj]
        del routes[ri]
        routes.append(new_route)
        # Update mapping
        for node in new_route[1:-1]:
            cust_to_route[node] = len(routes) - 1
    
    # If still too many routes, merge arbitrarily
    while len(routes) > truck_count:
        r1 = routes.pop(0)
        r2 = routes.pop(0)
        new_route = r1[:-1] + r2[1:]
        routes.append(new_route)
        for node in new_route[1:-1]:
            cust_to_route[node] = len(routes) - 1
    
    # Add empty routes if necessary
    while len(routes) < truck_count:
        routes.append([0, 0])
    
    # Helper to compute route length
    def route_length(route):
        total = 0.0
        for a, b in zip(route, route[1:]):
            total += distance_matrix[a, b]
        return total
    
    best_routes = [list(r) for r in routes]
    best_max = max(route_length(r) for r in best_routes)
    report_best_vrp(best_routes)
    
    # Local search
    max_iter = n * n
    for _ in range(max_iter):
        lengths = [route_length(r) for r in routes]
        max_len = max(lengths)
        longest_idx = lengths.index(max_len)
        longest_route = routes[longest_idx]
        improved = False
        # Try moving a customer from longest route to another
        for cust in longest_route[1:-1]:
            for other_idx, other_route in enumerate(routes):
                if other_idx == longest_idx:
                    continue
                new_long = [node for node in longest_route if node != cust]
                for pos in range(1, len(other_route)):
                    new_other = other_route[:pos] + [cust] + other_route[pos:]
                    new_long_len = route_length(new_long)
                    new_other_len = route_length(new_other)
                    other_lengths = [route_length(r) for idx, r in enumerate(routes) if idx not in (longest_idx, other_idx)]
                    new_max = max(new_long_len, new_other_len, max(other_lengths))
                    if new_max < best_max:
                        routes[longest_idx] = new_long
                        routes[other_idx] = new_other
                        best_max = new_max
                        best_routes = [list(r) for r in routes]
                        report_best_vrp(best_routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            # Try swapping customers between two routes
            for i in range(len(routes)):
                for j in range(i+1, len(routes)):
                    route_i = routes[i]
                    route_j = routes[j]
                    for ci in route_i[1:-1]:
                        for cj in route_j[1:-1]:
                            # Swap ci and cj
                            new_i = [0] + [cj if x == ci else x for x in route_i[1:-1]] + [0]
                            new_j = [0] + [ci if x == cj else x for x in route_j[1:-1]] + [0]
                            # Check no duplicate customers (should not happen as ci and cj are distinct)
                            len_i = route_length(new_i)
                            len_j = route_length(new_j)
                            other_lengths = [route_length(r) for idx, r in enumerate(routes) if idx not in (i, j)]
                            new_max = max(len_i, len_j, max(other_lengths))
                            if new_max < best_max:
                                routes[i] = new_i
                                routes[j] = new_j
                                best_max = new_max
                                best_routes = [list(r) for r in routes]
                                report_best_vrp(best_routes)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        if not improved:
            break
    return best_routes