import numpy as np
import heapq
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    num_customers = n - 1
    # Initial routes: each customer as a separate route, plus empty trucks if needed
    routes = [[0, i, 0] for i in range(1, n)]
    for _ in range(truck_count - num_customers):
        routes.append([0, 0])
    
    if num_customers >= truck_count:
        savings = []
        for i in range(1, n):
            for j in range(i+1, n):
                s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
                savings.append((-s, i, j))
        heapq.heapify(savings)
        
        customer_route = {i: idx for idx, route in enumerate(routes) for i in route[1:-1]}
        route_ends = {idx: [route[1], route[-2]] if len(route) > 2 else [] for idx, route in enumerate(routes)}
        
        while len(routes) > truck_count and savings:
            neg_s, i, j = heapq.heappop(savings)
            if i not in customer_route or j not in customer_route:
                continue
            ri = customer_route[i]
            rj = customer_route[j]
            if ri == rj:
                continue
            route_i = routes[ri]
            route_j = routes[rj]
            ends_i = route_ends[ri]
            ends_j = route_ends[rj]
            if i not in ends_i or j not in ends_j:
                continue
            inner_i = route_i[1:-1]
            inner_j = route_j[1:-1]
            combo = None
            if i == inner_i[-1] and j == inner_j[0]:
                combo = inner_i + inner_j
            elif i == inner_i[-1] and j == inner_j[-1]:
                combo = inner_i + inner_j[::-1]
            elif i == inner_i[0] and j == inner_j[0]:
                combo = inner_i[::-1] + inner_j
            elif i == inner_i[0] and j == inner_j[-1]:
                combo = inner_i[::-1] + inner_j[::-1]
            if combo is None:
                continue
            new_route = [0] + combo + [0]
            if ri < rj:
                routes.pop(rj)
                routes.pop(ri)
            else:
                routes.pop(ri)
                routes.pop(rj)
            routes.append(new_route)
            customer_route.clear()
            route_ends.clear()
            for idx, route in enumerate(routes):
                for c in route[1:-1]:
                    customer_route[c] = idx
                ends = [route[1], route[-2]] if len(route) > 2 else []
                route_ends[idx] = ends
    
    def route_distance(route):
        d = 0
        for a, b in zip(route, route[1:]):
            d += distance_matrix[a][b]
        return d
    
    best_routes = [list(r) for r in routes]
    best_max = max(route_distance(r) for r in best_routes)
    report_best_vrp(best_routes)
    
    max_iter = num_customers * truck_count * 2
    stagnation_limit = max(10, num_customers // 10)
    consecutive_no_improvement = 0
    
    for _ in range(max_iter):
        improved = False
        current_routes = [list(r) for r in best_routes]
        distances = [route_distance(r) for r in current_routes]
        max_idx = np.argmax(distances)
        min_idx = np.argmin(distances)
        if distances[max_idx] == distances[min_idx]:
            break
        best_delta = 0
        best_move = None
        longest_route = current_routes[max_idx]
        shortest_route = current_routes[min_idx]
        for i in range(1, len(longest_route)-1):
            cust = longest_route[i]
            new_long = longest_route[:i] + longest_route[i+1:]
            if len(new_long) == 2:
                new_long = [0,0]
            dist_long = route_distance(new_long)
            best_ins = None
            best_short_dist = None
            for ins in range(1, len(shortest_route)):
                new_short = shortest_route[:ins] + [cust] + shortest_route[ins:]
                d_short = route_distance(new_short)
                if best_short_dist is None or d_short < best_short_dist:
                    best_short_dist = d_short
                    best_ins = ins
            other_dists = [distances[k] for k in range(len(current_routes)) if k != max_idx and k != min_idx]
            new_max = max([dist_long, best_short_dist] + other_dists)
            # Use relative improvement threshold
            if new_max < distances[max_idx] * (1 - 1e-12):
                delta = distances[max_idx] - new_max
                if delta > best_delta:
                    best_delta = delta
                    best_move = (max_idx, min_idx, i, best_ins, cust, new_long, best_short_dist, other_dists)
        if best_move:
            max_idx, min_idx, i_remove, ins_pos, cust, new_long, new_short_dist, other = best_move
            new_routes = [list(r) for r in current_routes]
            new_routes[max_idx] = new_long
            new_routes[min_idx] = new_routes[min_idx][:ins_pos] + [cust] + new_routes[min_idx][ins_pos:]
            new_max = max([route_distance(r) for r in new_routes])
            if new_max < best_max * (1 - 1e-12):
                best_max = new_max
                best_routes = new_routes
                report_best_vrp(best_routes)
                improved = True
                consecutive_no_improvement = 0
        if not improved:
            consecutive_no_improvement += 1
            if consecutive_no_improvement >= stagnation_limit:
                break
    
    final_routes = [list(r) for r in best_routes]
    for i, route in enumerate(final_routes):
        if len(route) == 2:
            route = [0,0]
        else:
            if route[0] != 0:
                route.insert(0,0)
            if route[-1] != 0:
                route.append(0)
        final_routes[i] = route
    return final_routes