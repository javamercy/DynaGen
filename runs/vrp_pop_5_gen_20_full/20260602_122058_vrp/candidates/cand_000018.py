import numpy as np
import math
import random
import heapq
import itertools
import collections
import time

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    if truck_count >= n - 1:
        routes = [[0, i, 0] for i in range(1, n)]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes

    # Seed selection: farthest-first from depot
    seeds = []
    first_seed = max(range(1, n), key=lambda i: (distance_matrix[0][i], -i))
    seeds.append(first_seed)
    for _ in range(1, truck_count):
        best_min_dist = -1
        best_node = None
        for node in range(1, n):
            if node in seeds:
                continue
            min_dist = min(distance_matrix[node][s] for s in seeds)
            if min_dist > best_min_dist or (min_dist == best_min_dist and (best_node is None or node < best_node)):
                best_min_dist = min_dist
                best_node = node
        if best_node is None:
            break
        seeds.append(best_node)

    # Assign customers to nearest seed (tie by smaller seed index)
    clusters = {s: [] for s in seeds}
    for node in range(1, n):
        if node in seeds:
            clusters[node].append(node)
        else:
            nearest = min(seeds, key=lambda s: (distance_matrix[node][s], s))
            clusters[nearest].append(node)

    # Build initial routes using nearest neighbor from depot
    routes = []
    for seed in seeds:
        cluster = clusters[seed]
        unvisited = set(cluster)
        route = [0]
        current = 0
        while unvisited:
            next_node = min(unvisited, key=lambda x: (distance_matrix[current][x], x))
            route.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        route.append(0)
        routes.append(route)

    while len(routes) < truck_count:
        routes.append([0, 0])

    def route_dist(route):
        d = 0
        for i in range(len(route) - 1):
            d += distance_matrix[route[i]][route[i+1]]
        return d

    report_best_vrp(routes)

    # 2-opt improvement on each route
    max_iter = n
    for _ in range(max_iter):
        improved = False
        for idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            best_route = route[:]
            best_dist = route_dist(route)
            # Try all 2-opt swaps
            for i in range(1, len(route) - 2):
                for k in range(i+1, len(route) - 1):
                    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                    new_dist = route_dist(new_route)
                    if new_dist < best_dist:
                        best_dist = new_dist
                        best_route = new_route
            if best_dist < route_dist(route):
                routes[idx] = best_route
                improved = True
        if not improved:
            break
        # Call report after all routes improved (or after each improvement? we'll call after each full iteration)
        report_best_vrp(routes)
    return routes