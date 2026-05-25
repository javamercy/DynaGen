import numpy as np
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    customers.sort(key=lambda c: distance_matrix[0, c], reverse=True)
    
    def route_length(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total
    
    def best_insertion(route, customer):
        best_pos = -1
        best_inc = float('inf')
        for i in range(1, len(route)):
            prev, nxt = route[i-1], route[i]
            inc = distance_matrix[prev, customer] + distance_matrix[customer, nxt] - distance_matrix[prev, nxt]
            if inc < best_inc:
                best_inc = inc
                best_pos = i
        return best_pos, best_inc
    
    def construct():
        routes = [[0, 0] for _ in range(truck_count)]
        lengths = [0.0] * truck_count
        for cust in customers:
            best_route = -1
            best_pos = -1
            best_new_max = float('inf')
            for r_idx in range(truck_count):
                route = routes[r_idx]
                pos, inc = best_insertion(route, cust)
                new_len = lengths[r_idx] + inc
                other_max = max(lengths[:r_idx] + lengths[r_idx+1:]) if truck_count > 1 else 0.0
                new_max = max(other_max, new_len)
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_route = r_idx
                    best_pos = pos
            route = routes[best_route]
            route.insert(best_pos, cust)
            lengths[best_route] = route_length(route)
        return routes, lengths
    
    def two_opt(routes, lengths):
        for r_idx in range(truck_count):
            route = routes[r_idx]
            improved = True
            max_iter = len(route) * len(route)
            for _ in range(max_iter):
                improved = False
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        if j - i == 1:
                            continue
                        old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                        new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                        if new < old - 1e-12:
                            route[i:j+1] = reversed(route[i:j+1])
                            improved = True
                if not improved:
                    break
            lengths[r_idx] = route_length(route)
        return routes, lengths
    
    def relocate(routes, lengths):
        improved = True
        max_iters = n * n
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            current_max = max(lengths)
            for c in customers:
                for r_idx, route in enumerate(routes):
                    if c in route:
                        break
                old_route = route
                old_len = lengths[r_idx]
                new_route = [x for x in old_route if x != c]
                new_len = route_length(new_route)
                for r2_idx, r2 in enumerate(routes):
                    if r2_idx == r_idx:
                        continue
                    for pos in range(1, len(r2)):
                        # insertion cost
                        inc = distance_matrix[r2[pos-1], c] + distance_matrix[c, r2[pos]] - distance_matrix[r2[pos-1], r2[pos]]
                        new_len2 = lengths[r2_idx] + inc
                        new_max = max(lengths[:r_idx] + [new_len] + lengths[r_idx+1:r2_idx] + [new_len2] + lengths[r2_idx+1:])
                        if new_max < current_max:
                            # apply move
                            routes[r_idx] = new_route
                            routes[r2_idx] = r2[:pos] + [c] + r2[pos:]
                            lengths[r_idx] = new_len
                            lengths[r2_idx] = new_len2
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes, lengths
    
    best_max = float('inf')
    best_routes = None
    visited_sigs = set()
    
    for restart in range(10):
        if restart == 0:
            routes, lengths = construct()
        else:
            if best_routes is None:
                continue
            # copy best
            routes = [r[:] for r in best_routes]
            lengths = [route_length(r) for r in routes]
            # identify longest route
            longest_idx = int(np.argmax(lengths))
            longest_route = routes[longest_idx]
            # remove 3 customers from longest route (if possible)
            remove_count = min(3, len(longest_route)-2)
            removed = []
            for _ in range(remove_count):
                if len(longest_route) > 2:
                    c = longest_route.pop(1)
                    removed.append(c)
            lengths[longest_idx] = route_length(longest_route)
            # reinsert using regret-based min-max
            customers_to_insert = removed[:]
            while customers_to_insert:
                best_cust = None
                best_regret = -float('inf')
                best_route_idx = None
                best_pos = None
                for cust in customers_to_insert:
                    best_inc = float('inf')
                    best_route_c = -1
                    best_pos_c = -1
                    second_best = float('inf')
                    for r_idx, route in enumerate(routes):
                        pos, inc = best_insertion(route, cust)
                        if inc < best_inc:
                            second_best = best_inc
                            best_inc = inc
                            best_route_c = r_idx
                            best_pos_c = pos
                        elif inc < second_best:
                            second_best = inc
                    regret = second_best - best_inc
                    if regret > best_regret:
                        best_regret = regret
                        best_cust = cust
                        best_route_idx = best_route_c
                        best_pos = best_pos_c
                # insert best customer
                routes[best_route_idx].insert(best_pos, best_cust)
                lengths[best_route_idx] = route_length(routes[best_route_idx])
                customers_to_insert.remove(best_cust)
        # improvement
        routes, lengths = two_opt(routes, lengths)
        routes, lengths = relocate(routes, lengths)
        # check diversity and update best
        sig = tuple(sorted(tuple(r) for r in routes))
        if sig not in visited_sigs:
            visited_sigs.add(sig)
            max_len = max(lengths)
            if max_len < best_max:
                best_max = max_len
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
    
    if best_routes is None:
        best_routes = routes
    return best_routes