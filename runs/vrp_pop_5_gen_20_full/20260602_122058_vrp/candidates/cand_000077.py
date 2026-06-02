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
        best_min_dist = -1.0
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
        d = 0.0
        for i in range(len(route) - 1):
            d += distance_matrix[route[i]][route[i+1]]
        return d

    def max_route_dist(routes):
        return max(route_dist(r) for r in routes)

    best_routes = [r[:] for r in routes]
    best_max = max_route_dist(routes)

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
        current_max = max_route_dist(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
        report_best_vrp(routes)

    # Cross-route improvement targeting the longest route
    for _ in range(n):
        # Find longest route
        longest_idx = max(range(len(routes)), key=lambda idx: route_dist(routes[idx]))
        longest_route = routes[longest_idx]
        if len(longest_route) <= 3:
            break
        improved = False
        # Try relocating each node from longest route to other routes
        for node_pos in range(1, len(longest_route)-1):
            node = longest_route[node_pos]
            new_route_long = longest_route[:node_pos] + longest_route[node_pos+1:]
            dist_long = route_dist(new_route_long)
            for other_idx, other_route in enumerate(routes):
                if other_idx == longest_idx:
                    continue
                # Find best insertion position in other_route
                best_other = None
                best_other_dist = float('inf')
                for insert_pos in range(1, len(other_route)):
                    new_other = other_route[:insert_pos] + [node] + other_route[insert_pos:]
                    d_other = route_dist(new_other)
                    if d_other < best_other_dist:
                        best_other_dist = d_other
                        best_other = new_other
                if best_other is None:
                    continue
                # Evaluate new max
                new_max = max(dist_long, best_other_dist)
                for k_idx, k_route in enumerate(routes):
                    if k_idx not in (longest_idx, other_idx):
                        new_max = max(new_max, route_dist(k_route))
                if new_max < best_max:
                    # Accept move
                    routes[longest_idx] = new_route_long
                    routes[other_idx] = best_other
                    best_max = new_max
                    best_routes = [r[:] for r in routes]
                    improved = True
                    break
            if improved:
                break
        if improved:
            report_best_vrp(routes)
            continue
        # Try swapping a node from longest route with a node from another route
        for node_pos in range(1, len(longest_route)-1):
            node_a = longest_route[node_pos]
            for other_idx, other_route in enumerate(routes):
                if other_idx == longest_idx or len(other_route) <= 3:
                    continue
                for other_pos in range(1, len(other_route)-1):
                    node_b = other_route[other_pos]
                    # Create new routes after swap
                    new_long = longest_route[:node_pos] + [node_b] + longest_route[node_pos+1:]
                    new_other = other_route[:other_pos] + [node_a] + other_route[other_pos+1:]
                    dist_long = route_dist(new_long)
                    dist_other = route_dist(new_other)
                    new_max = max(dist_long, dist_other)
                    for k_idx, k_route in enumerate(routes):
                        if k_idx not in (longest_idx, other_idx):
                            new_max = max(new_max, route_dist(k_route))
                    if new_max < best_max:
                        routes[longest_idx] = new_long
                        routes[other_idx] = new_other
                        best_max = new_max
                        best_routes = [r[:] for r in routes]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            report_best_vrp(routes)
        else:
            break

    # Restore best found
    routes = best_routes
    report_best_vrp(routes)
    return routes