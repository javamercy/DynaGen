import numpy as np
import math
import random
from itertools import combinations, permutations
from collections import deque

def report_best_vrp(routes):
    pass

def total_distance(route, dist):
    d = 0
    for i in range(len(route)-1):
        d += dist[route[i]][route[i+1]]
    return d

def max_route_distance(routes, dist):
    return max(total_distance(r, dist) for r in routes)

def initial_solution(distance_matrix, n, m):
    routes = [[0,0] for _ in range(m)]
    assigned = [False]*n
    assigned[0]=True
    customers = list(range(1,n))
    # Greedy: assign farthest unassigned customer to route that minimizes increase in max distance
    while customers:
        # find farthest unassigned customer from depot
        farthest = max(customers, key=lambda c: distance_matrix[0][c])
        best_route = -1
        best_inc = float('inf')
        for i in range(m):
            if len(routes[i]) == 2:  # only [0,0]
                new_route = [0, farthest, 0]
                new_max = total_distance(new_route, distance_matrix)
            else:
                # try inserting farthest at best position to minimize route increase
                best_local = float('inf')
                best_pos = -1
                for pos in range(1, len(routes[i])):
                    new_route = routes[i][:pos] + [farthest] + routes[i][pos:]
                    d = total_distance(new_route, distance_matrix)
                    if d < best_local:
                        best_local = d
                        best_pos = pos
                # new route after insertion
                new_route = routes[i][:best_pos] + [farthest] + routes[i][best_pos:]
                new_dist = best_local
            # compute new max distance if we assign to route i
            current_max = max_route_distance(routes, distance_matrix)
            # we need to compute the new max after replacing route i
            # simpler: compute max of all routes after insertion
            # for speed, compute potential new max
            new_routes = [list(r) for r in routes]
            # simulate insertion
            if len(routes[i]) == 2:
                new_routes[i] = [0, farthest, 0]
            else:
                new_routes[i] = routes[i][:best_pos] + [farthest] + routes[i][best_pos:]
            new_max = total_distance(new_routes[i], distance_matrix)
            other_max = 0
            for j in range(m):
                if j != i:
                    other_max = max(other_max, total_distance(routes[j], distance_matrix))
            max_inc = max(new_max, other_max)
            if max_inc < best_inc:
                best_inc = max_inc
                best_route = i
        # assign farthest to best route
        routes[best_route] = (routes[best_route][:-1] + [farthest] + [0] if len(routes[best_route])==2 else
                              routes[best_route][:best_pos] + [farthest] + routes[best_route][best_pos:])
        customers.remove(farthest)
        assigned[farthest]=True
    return routes

def two_opt_route(route, dist):
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route)-2):
            for j in range(i+1, len(route)-1):
                if j == i+1:
                    continue
                old_dist = dist[route[i-1]][route[i]] + dist[route[j]][route[j+1]]
                new_dist = dist[route[i-1]][route[j]] + dist[route[i]][route[j+1]]
                if new_dist < old_dist:
                    route[i:j+1] = reversed(route[i:j+1])
                    improved = True
    return route

def relocate_move(routes, dist, n, m):
    # find longest route
    max_len_route = max(range(m), key=lambda i: total_distance(routes[i], dist))
    best_improvement = 0
    best_move = None
    if len(routes[max_len_route]) <= 2:
        return False
    # try moving each customer in longest route to another route
    for cust in routes[max_len_route][1:-1]:
        for other_route in range(m):
            if other_route == max_len_route:
                continue
            # compute distance increase for other route if we insert cust at best position
            for pos in range(1, len(routes[other_route])):
                # simulate insertion
                new_other = routes[other_route][:pos] + [cust] + routes[other_route][pos:]
                d_other = total_distance(new_other, dist)
                # remove cust from longest route
                new_long = [c for c in routes[max_len_route] if c != cust]
                d_long = total_distance(new_long, dist)
                # ensure new_long starts and ends with 0
                if new_long[0] != 0 or new_long[-1] != 0:
                    continue
                current_max = max_route_distance(routes, dist)
                new_max = max(d_long, d_other, max(total_distance(routes[i], dist) for i in range(m) if i not in {max_len_route, other_route}))
                if new_max < current_max:
                    improvement = current_max - new_max
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = (max_len_route, cust, other_route, pos, new_long, new_other)
    if best_move:
        max_len_route, cust, other_route, pos, new_long, new_other = best_move
        routes[max_len_route] = new_long
        routes[other_route] = new_other
        return True
    return False

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    m = truck_count
    routes = initial_solution(distance_matrix, n, m)
    report_best_vrp(routes)
    # Local search
    improved = True
    max_iter = n * n
    it = 0
    while improved and it < max_iter:
        improved = False
        # intra-route 2-opt on all routes
        for i in range(m):
            if len(routes[i]) > 3:
                old_dist = total_distance(routes[i], distance_matrix)
                routes[i] = two_opt_route(routes[i], distance_matrix)
                if total_distance(routes[i], distance_matrix) < old_dist:
                    improved = True
        # inter-route relocate
        if relocate_move(routes, distance_matrix, n, m):
            improved = True
        # report if better
        if improved:
            report_best_vrp(routes)
        it += 1
    return routes