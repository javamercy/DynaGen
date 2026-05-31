import random
import numpy as np
from math import inf

def solve_vrp(distance_matrix, truck_count):
    random.seed(0)
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

    def route_length(route):
        total = 0
        for i in range(len(route) - 1):
            total += distance_matrix[route[i]][route[i+1]]
        return total

    def two_opt(route, max_iter=10):
        route = route[:]
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_length(new_route) < route_length(route):
                        route = new_route
                        improved = True
        return route

    def split_tour(tour, L):
        """Greedy split: return list of routes if possible with max route distance <= L, else None"""
        routes = []
        current_route = [0]
        current_dist = 0.0
        for node in tour:
            # distance from current position to node
            if len(current_route) == 1:  # just depot
                dist_to_node = distance_matrix[0][node]
            else:
                dist_to_node = distance_matrix[current_route[-1]][node]
            if current_dist + dist_to_node + distance_matrix[node][0] > L:
                # close current route
                current_route.append(0)
                routes.append(current_route)
                # start new route
                current_route = [0]
                current_dist = 0.0
                # now add node to new route
                dist_to_node = distance_matrix[0][node]
            current_route.append(node)
            current_dist += dist_to_node
        # finish last route
        if len(current_route) > 1:
            current_route.append(0)
            routes.append(current_route)
        else:
            # empty route
            routes.append([0, 0])
        if len(routes) > truck_count:
            return None
        # pad with empty routes if needed
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def binary_search_split(tour):
        total_dist = 0
        for i in range(len(tour)):
            if i == 0:
                total_dist += distance_matrix[0][tour[0]] + distance_matrix[tour[-1]][0]
            else:
                total_dist += distance_matrix[tour[i-1]][tour[i]]
        total_dist = route_length([0] + tour + [0])
        low = max( (distance_matrix[0][tour[0]] + distance_matrix[tour[-1]][0]) if len(tour) > 0 else 0,
                   max(distance_matrix[0][c] + distance_matrix[c][0] for c in tour) ) if tour else 0
        high = total_dist
        best_routes = None
        while high - low > 1e-6:
            mid = (low + high) / 2
            routes = split_tour(tour, mid)
            if routes is not None:
                best_routes = routes
                high = mid
            else:
                low = mid
        # refine with low bound
        routes = split_tour(tour, low)
        if routes is not None:
            best_routes = routes
        return best_routes

    best_max = float('inf')
    best_routes = None
    num_restarts = min(20, max(1, n // 5))
    for _ in range(num_restarts):
        # generate random permutation of customers
        perm = customers[:]
        random.shuffle(perm)
        # split
        routes = binary_search_split(perm)
        if routes is None:
            continue
        # apply 2-opt to each route
        for i in range(truck_count):
            if len(routes[i]) > 2:
                routes[i] = two_opt(routes[i], max_iter=len(routes[i]) * 2)
        # compute lengths
        lengths = [route_length(r) for r in routes]
        # local search: relocate best move
        improved = True
        max_iter_local = n * truck_count
        it_local = 0
        while improved and it_local < max_iter_local:
            improved = False
            it_local += 1
            # find longest route
            long_idx = max(range(truck_count), key=lambda i: lengths[i])
            long_route = routes[long_idx]
            if len(long_route) <= 2:
                break
            best_move = None
            best_new_max = lengths[long_idx]
            for pos in range(1, len(long_route) - 1):
                cust = long_route[pos]
                # remove cust from long route
                new_long = long_route[:pos] + long_route[pos+1:]
                new_long_len = route_length(new_long)
                for target_idx in range(truck_count):
                    if target_idx == long_idx:
                        continue
                    target_route = routes[target_idx]
                    # try all insertion positions
                    for ins_pos in range(1, len(target_route)):
                        new_target = target_route[:ins_pos] + [cust] + target_route[ins_pos:]
                        new_target_len = route_length(new_target)
                        new_max = max(new_long_len, new_target_len, max(lengths[i] for i in range(truck_count) if i not in [long_idx, target_idx]))
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = (long_idx, target_idx, cust, ins_pos, new_long, new_target, new_long_len, new_target_len)
            if best_move is not None and best_new_max < lengths[long_idx]:
                long_idx, target_idx, cust, ins_pos, new_long, new_target, new_long_len, new_target_len = best_move
                routes[long_idx] = new_long
                routes[target_idx] = new_target
                lengths[long_idx] = new_long_len
                lengths[target_idx] = new_target_len
                improved = True
                # update best
                current_max = max(lengths)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
        # evaluate final solution
        current_max = max(lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
    return best_routes