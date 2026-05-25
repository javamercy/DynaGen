import numpy as np
import math
import random
import heapq
import itertools
from collections import defaultdict

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    depot = 0
    routes = [[depot, depot] for _ in range(truck_count)]
    unassigned = set(range(1, n))

    def route_distance(route):
        dist = 0.0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i], route[i+1]]
        return dist

    def best_insertion(route, customer):
        best_pos = -1
        best_delta = float('inf')
        for pos in range(1, len(route)):
            prev = route[pos-1]
            nxt = route[pos]
            delta = distance_matrix[prev, customer] + distance_matrix[customer, nxt] - distance_matrix[prev, nxt]
            if delta < best_delta:
                best_delta = delta
                best_pos = pos
        return best_pos, best_delta

    while unassigned:
        cust_info = []
        for cust in unassigned:
            deltas = []
            positions = []
            for ridx, route in enumerate(routes):
                pos, delta = best_insertion(route, cust)
                deltas.append(delta)
                positions.append(pos)
            sorted_deltas = sorted(enumerate(deltas), key=lambda x: x[1])
            best_ridx, best_delta = sorted_deltas[0]
            second_best_delta = sorted_deltas[1][1] if len(sorted_deltas) > 1 else best_delta
            regret = second_best_delta - best_delta
            cust_info.append((regret, best_delta, cust, best_ridx, positions[best_ridx]))
        cust_info.sort(key=lambda x: (-x[0], x[1], x[2]))
        _, _, cust, ridx, pos = cust_info[0]
        route = routes[ridx]
        route.insert(pos, cust)
        unassigned.remove(cust)

    route_dists = [route_distance(r) for r in routes]
    best_routes = [r[:] for r in routes]
    best_max = max(route_dists)
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass

    max_iter = 10 * n
    for _ in range(max_iter):
        current_max = max(route_dists)
        max_idx = route_dists.index(current_max)
        improved = False

        route = routes[max_idx]
        for i in range(1, len(route)-1):
            cust = route[i]
            new_route = route[:i] + route[i+1:]
            new_dist = route_distance(new_route)
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                for pos in range(1, len(other_route)):
                    new_other = other_route[:pos] + [cust] + other_route[pos:]
                    new_other_dist = route_distance(new_other)
                    candidate_max = max(new_dist, new_other_dist)
                    if candidate_max < current_max:
                        routes[max_idx] = new_route
                        routes[other_idx] = new_other
                        route_dists[max_idx] = new_dist
                        route_dists[other_idx] = new_other_dist
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            route = routes[max_idx]
            best_route = route[:]
            best_dist = route_dists[max_idx]
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dist:
                        best_route = new_route
                        best_dist = new_dist
            if best_dist < route_dists[max_idx]:
                routes[max_idx] = best_route
                route_dists[max_idx] = best_dist
                improved = True

        if improved:
            new_max = max(route_dists)
            if new_max < current_max:
                best_routes = [r[:] for r in routes]
                try:
                    report_best_vrp(best_routes)
                except NameError:
                    pass
        else:
            break

    while len(best_routes) < truck_count:
        best_routes.append([depot, depot])
    return best_routes