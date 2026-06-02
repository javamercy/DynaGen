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

    def max_route_dist(routes):
        return max(route_dist(r) for r in routes)

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
        report_best_vrp(routes)

    # Cross-route improvement: relocate from longest route
    for _ in range(n):
        # Find longest route (by distance)
        longest_idx = max(range(len(routes)), key=lambda idx: route_dist(routes[idx]))
        longest_route = routes[longest_idx]
        if len(longest_route) <= 3:
            break
        current_max = max_route_dist(routes)
        best_move = None
        best_new_max = current_max
        # Consider each customer in longest route (excluding depots)
        for pos in range(1, len(longest_route) - 1):
            customer = longest_route[pos]
            # Try moving to each other route at every insertion position
            for other_idx in range(len(routes)):
                if other_idx == longest_idx:
                    continue
                other_route = routes[other_idx]
                # Insert after depot or between any two nodes
                for insert_pos in range(1, len(other_route)):
                    # Build new routes
                    new_longest = longest_route[:pos] + longest_route[pos+1:]
                    new_other = other_route[:insert_pos] + [customer] + other_route[insert_pos:]
                    # Ensure no extra depot? other_route already has depots at ends
                    if len(new_longest) == 2:  # only depot left
                        new_longest = [0, 0]
                    new_routes = routes[:]
                    new_routes[longest_idx] = new_longest
                    new_routes[other_idx] = new_other
                    new_max = max(route_dist(r) for r in new_routes)
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_move = (longest_idx, pos, other_idx, insert_pos)
        if best_move is not None and best_new_max < current_max:
            long_idx, pos, other_idx, ins_pos = best_move
            customer = routes[long_idx][pos]
            new_longest = routes[long_idx][:pos] + routes[long_idx][pos+1:]
            if len(new_longest) == 2:
                new_longest = [0, 0]
            new_other = routes[other_idx][:ins_pos] + [customer] + routes[other_idx][ins_pos:]
            routes[long_idx] = new_longest
            routes[other_idx] = new_other
            report_best_vrp(routes)
        else:
            break

    return routes