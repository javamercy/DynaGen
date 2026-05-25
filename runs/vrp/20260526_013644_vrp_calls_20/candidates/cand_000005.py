import numpy as np
import math
import random
import heapq
import itertools
import collections
import time

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    routes = [[0,0] for _ in range(truck_count)]
    route_distances = [0.0 for _ in range(truck_count)]

    def route_distance(route):
        if len(route) == 2:
            return 0.0
        dist = 0.0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i], route[i+1]]
        return dist

    sorted_customers = sorted(customers, key=lambda c: distance_matrix[0,c], reverse=True)

    for cust in sorted_customers:
        best_max = float('inf')
        best_route_idx = None
        best_pos = None
        best_new_dist = None
        for r in range(truck_count):
            route = routes[r]
            for pos in range(1, len(route)):
                delta = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                new_route_dist = route_distances[r] + delta
                new_max = new_route_dist
                for other_r in range(truck_count):
                    if other_r != r:
                        if route_distances[other_r] > new_max:
                            new_max = route_distances[other_r]
                if new_max < best_max or (new_max == best_max and route_distances[r] < route_distances[best_route_idx]):
                    best_max = new_max
                    best_route_idx = r
                    best_pos = pos
                    best_new_dist = new_route_dist
        routes[best_route_idx].insert(best_pos, cust)
        route_distances[best_route_idx] = best_new_dist

    def compute_max_distance(routes):
        max_dist = 0.0
        for r in routes:
            d = route_distance(r)
            if d > max_dist:
                max_dist = d
        return max_dist

    current_max = compute_max_distance(routes)
    best_max = current_max
    best_routes = [r[:] for r in routes]

    try:
        report_best_vrp(best_routes)
    except NameError:
        pass

    max_iterations = 1000
    iteration = 0
    improved = True
    while improved and iteration < max_iterations:
        improved = False
        # Intra-route 2-opt
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 3:
                continue
            best_delta = 0.0
            best_i = None
            best_j = None
            for i in range(1, len(route)-2):
                for j in range(i+2, len(route)-1):
                    delta = distance_matrix[route[i], route[j]] + distance_matrix[route[i+1], route[j+1]] - distance_matrix[route[i], route[i+1]] - distance_matrix[route[j], route[j+1]]
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            if best_delta < 0:
                new_route = route[:best_i+1] + route[best_j:best_i:-1] + route[best_j+1:]
                routes[r_idx] = new_route
                route_distances[r_idx] = route_distance(new_route)
                current_max = compute_max_distance(routes)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    try:
                        report_best_vrp(best_routes)
                    except NameError:
                        pass
                improved = True

        # Inter-route relocate
        best_move = None
        best_new_max = current_max
        for r_from in range(truck_count):
            if len(routes[r_from]) <= 2:
                continue
            for pos_from in range(1, len(routes[r_from])-1):
                cust = routes[r_from][pos_from]
                prev = routes[r_from][pos_from-1]
                next = routes[r_from][pos_from+1]
                delta_from = distance_matrix[prev, next] - distance_matrix[prev, cust] - distance_matrix[cust, next]
                new_from_dist = route_distances[r_from] + delta_from
                for r_to in range(truck_count):
                    if r_to == r_from:
                        continue
                    route_to = routes[r_to]
                    for pos_to in range(1, len(route_to)):
                        prev_to = route_to[pos_to-1]
                        next_to = route_to[pos_to]
                        delta_to = distance_matrix[prev_to, cust] + distance_matrix[cust, next_to] - distance_matrix[prev_to, next_to]
                        new_to_dist = route_distances[r_to] + delta_to
                        cand_max = max(new_from_dist, new_to_dist)
                        for other_r in range(truck_count):
                            if other_r != r_from and other_r != r_to:
                                cand_max = max(cand_max, route_distances[other_r])
                        if cand_max < best_new_max:
                            best_new_max = cand_max
                            best_move = (r_from, pos_from, r_to, pos_to, new_from_dist, new_to_dist)
        if best_move is not None:
            r_from, pos_from, r_to, pos_to, new_from_dist, new_to_dist = best_move
            cust = routes[r_from].pop(pos_from)
            routes[r_to].insert(pos_to, cust)
            route_distances[r_from] = new_from_dist
            route_distances[r_to] = new_to_dist
            current_max = best_new_max
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in routes]
                try:
                    report_best_vrp(best_routes)
                except NameError:
                    pass
            improved = True

        # Inter-route swap
        best_move = None
        best_new_max = current_max
        for r1 in range(truck_count):
            if len(routes[r1]) <= 2:
                continue
            for pos1 in range(1, len(routes[r1])-1):
                cust1 = routes[r1][pos1]
                prev1 = routes[r1][pos1-1]
                next1 = routes[r1][pos1+1]
                for r2 in range(r1+1, truck_count):
                    if len(routes[r2]) <= 2:
                        continue
                    for pos2 in range(1, len(routes[r2])-1):
                        cust2 = routes[r2][pos2]
                        prev2 = routes[r2][pos2-1]
                        next2 = routes[r2][pos2+1]
                        delta1 = distance_matrix[prev1,cust2] + distance_matrix[cust2,next1] - distance_matrix[prev1,cust1] - distance_matrix[cust1,next1]
                        new_dist1 = route_distances[r1] + delta1
                        delta2 = distance_matrix[prev2,cust1] + distance_matrix[cust1,next2] - distance_matrix[prev2,cust2] - distance_matrix[cust2,next2]
                        new_dist2 = route_distances[r2] + delta2
                        cand_max = max(new_dist1, new_dist2)
                        for other_r in range(truck_count):
                            if other_r != r1 and other_r != r2:
                                cand_max = max(cand_max, route_distances[other_r])
                        if cand_max < best_new_max:
                            best_new_max = cand_max
                            best_move = (r1, pos1, r2, pos2, new_dist1, new_dist2)
        if best_move is not None:
            r1, pos1, r2, pos2, new_dist1, new_dist2 = best_move
            cust1 = routes[r1][pos1]
            cust2 = routes[r2][pos2]
            routes[r1][pos1] = cust2
            routes[r2][pos2] = cust1
            route_distances[r1] = new_dist1
            route_distances[r2] = new_dist2
            current_max = best_new_max
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in routes]
                try:
                    report_best_vrp(best_routes)
                except NameError:
                    pass
            improved = True

        iteration += 1

    return best_routes