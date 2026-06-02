import numpy as np
import math
import random
import heapq
import itertools
import collections

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    if truck_count >= n - 1:
        routes = [[0, i, 0] for i in range(1, n)]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes

    # farthest-first seed selection
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

    # assign customers to nearest seed (tie by smaller seed index)
    clusters = {s: [] for s in seeds}
    for node in range(1, n):
        if node in seeds:
            clusters[node].append(node)
        else:
            nearest = min(seeds, key=lambda s: (distance_matrix[node][s], s))
            clusters[nearest].append(node)

    # build initial routes using nearest neighbor from depot
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

    def total_dist(routes):
        return sum(route_dist(r) for r in routes)

    def max_dist(routes):
        return max(route_dist(r) for r in routes)

    def route_dist_after_move(route, i, j, move_type):
        # compute distance change for a move on route
        # move_type: '2opt', 'oropt' (i to j reversed or moved sequence)
        # This is simplified; for full evaluation we compute new route distance
        pass

    best_routes = [r[:] for r in routes]
    best_max = max_dist(routes)
    report_best_vrp(routes)

    # Tabu search parameters
    tabu_tenure = max(5, n // 5)
    tabu_list = {}
    max_iter = n * truck_count * 2
    no_improve_iter = 0
    restart_limit = max(10, n // 2)

    for iteration in range(max_iter):
        # neighborhoods
        best_move = None
        best_move_type = None
        best_new_max = float('inf')
        best_new_total = float('inf')
        best_route_idx = None
        best_route_old = None
        best_route_new = None

        # intra-route: 2-opt and Or-opt
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            # 2-opt
            for i in range(1, len(route) - 2):
                for k in range(i+1, len(route) - 1):
                    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                    new_max = max(max_dist(routes[:r_idx] + [new_route] + routes[r_idx+1:]), 0)
                    new_total = total_dist(routes[:r_idx] + [new_route] + routes[r_idx+1:])
                    move_key = (r_idx, '2opt', i, k)
                    if move_key not in tabu_list or new_max < best_max:
                        if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                            best_new_max = new_max
                            best_new_total = new_total
                            best_move = move_key
                            best_move_type = '2opt'
                            best_route_idx = r_idx
                            best_route_old = route[:]
                            best_route_new = new_route
            # Or-opt (relocate a sequence of 1-3 nodes)
            for seq_len in range(1, 4):
                for i in range(1, len(route) - seq_len):
                    seq = route[i:i+seq_len]
                    # remove seq from route
                    temp_route = route[:i] + route[i+seq_len:]
                    # insert at each position from 1 to len(temp_route)-1
                    for pos in range(1, len(temp_route)):
                        new_route = temp_route[:pos] + seq + temp_route[pos:]
                        # ensure it's different
                        if new_route == route:
                            continue
                        new_max = max(max_dist(routes[:r_idx] + [new_route] + routes[r_idx+1:]), 0)
                        new_total = total_dist(routes[:r_idx] + [new_route] + routes[r_idx+1:])
                        move_key = (r_idx, 'oropt', i, seq_len, pos)
                        if move_key not in tabu_list or new_max < best_max:
                            if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                                best_new_max = new_max
                                best_new_total = new_total
                                best_move = move_key
                                best_move_type = 'oropt'
                                best_route_idx = r_idx
                                best_route_old = route[:]
                                best_route_new = new_route

        # inter-route: relocate and swap
        for r1 in range(len(routes)):
            for r2 in range(len(routes)):
                if r1 >= r2:
                    continue
                route1 = routes[r1]
                route2 = routes[r2]
                # relocate node from route1 to route2
                for node_pos in range(1, len(route1)-1):
                    node = route1[node_pos]
                    new_route1 = route1[:node_pos] + route1[node_pos+1:]
                    for ins_pos in range(1, len(route2)):
                        new_route2 = route2[:ins_pos] + [node] + route2[ins_pos:]
                        new_routes = routes[:]
                        new_routes[r1] = new_route1
                        new_routes[r2] = new_route2
                        new_max = max(max_dist(new_routes), 0)
                        new_total = total_dist(new_routes)
                        move_key = (r1, r2, 'relocate', node_pos, ins_pos)
                        if move_key not in tabu_list or new_max < best_max:
                            if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                                best_new_max = new_max
                                best_new_total = new_total
                                best_move = move_key
                                best_move_type = 'relocate'
                                best_route_idx = (r1, r2)
                                best_route_old = (route1[:], route2[:])
                                best_route_new = (new_route1, new_route2)
                # swap nodes between routes
                for pos1 in range(1, len(route1)-1):
                    node1 = route1[pos1]
                    for pos2 in range(1, len(route2)-1):
                        node2 = route2[pos2]
                        new_route1 = route1[:]
                        new_route1[pos1] = node2
                        new_route2 = route2[:]
                        new_route2[pos2] = node1
                        new_routes = routes[:]
                        new_routes[r1] = new_route1
                        new_routes[r2] = new_route2
                        new_max = max(max_dist(new_routes), 0)
                        new_total = total_dist(new_routes)
                        move_key = (r1, r2, 'swap', pos1, pos2)
                        if move_key not in tabu_list or new_max < best_max:
                            if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                                best_new_max = new_max
                                best_new_total = new_total
                                best_move = move_key
                                best_move_type = 'swap'
                                best_route_idx = (r1, r2)
                                best_route_old = (route1[:], route2[:])
                                best_route_new = (new_route1, new_route2)

        if best_move is None:
            break

        # apply move
        if best_move_type in ('2opt', 'oropt'):
            routes[best_route_idx] = best_route_new
        else:
            r1, r2 = best_route_idx
            routes[r1], routes[r2] = best_route_new

        # update tabu list
        tabu_list[best_move] = iteration + tabu_tenure
        # remove expired tabu entries
        expired = [k for k, v in tabu_list.items() if v <= iteration]
        for k in expired:
            del tabu_list[k]

        # check best
        cur_max = max_dist(routes)
        if cur_max < best_max:
            best_max = cur_max
            best_routes = [r[:] for r in routes]
            no_improve_iter = 0
            report_best_vrp(routes)
        else:
            no_improve_iter += 1

        # restart if no improvement for restart_limit iterations
        if no_improve_iter >= restart_limit:
            # perturbation: reassign a few customers to different routes
            # relocate random move to escape
            # simple: take a random node from the longest route and move to shortest route (or nearest seed)
            longest_route_idx = max(range(len(routes)), key=lambda i: route_dist(routes[i]))
            longest_route = routes[longest_route_idx]
            if len(longest_route) > 3:
                # pick a random customer
                node_pos = random.randint(1, len(longest_route)-2)
                node = longest_route[node_pos]
                # find best insertion in another route to minimize max
                best_insert = None
                best_new_max = float('inf')
                for r_idx, route in enumerate(routes):
                    if r_idx == longest_route_idx:
                        continue
                    for ins_pos in range(1, len(route)):
                        new_route1 = longest_route[:node_pos] + longest_route[node_pos+1:]
                        new_route2 = route[:ins_pos] + [node] + route[ins_pos:]
                        temp_routes = routes[:]
                        temp_routes[longest_route_idx] = new_route1
                        temp_routes[r_idx] = new_route2
                        new_max = max_dist(temp_routes)
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_insert = (r_idx, ins_pos)
                if best_insert:
                    r_idx, ins_pos = best_insert
                    new_route1 = routes[longest_route_idx][:node_pos] + routes[longest_route_idx][node_pos+1:]
                    new_route2 = routes[r_idx][:ins_pos] + [node] + routes[r_idx][ins_pos:]
                    routes[longest_route_idx] = new_route1
                    routes[r_idx] = new_route2
                    no_improve_iter = 0
                    # clear tabu list partially
                    tabu_list.clear()
                    report_best_vrp(routes)
            else:
                # if cannot perturb, reduce restart limit further
                restart_limit = max(5, restart_limit - 1)

    # ensure best routes returned
    routes = best_routes
    report_best_vrp(routes)
    return routes