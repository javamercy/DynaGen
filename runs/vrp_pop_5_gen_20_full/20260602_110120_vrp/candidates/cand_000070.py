import numpy as np
import random
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    random.seed(0)
    
    def compute_route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def construction():
        routes = [[0, 0] for _ in range(truck_count)]
        route_distances = [0.0 for _ in range(truck_count)]
        unassigned = set(range(1, n))
        while unassigned:
            best_customer = None
            best_max_val = float('inf')
            best_pos = None
            for c in unassigned:
                for r_idx, route in enumerate(routes):
                    curr_dist = route_distances[r_idx]
                    for i in range(1, len(route)):
                        new_dist = curr_dist - distance_matrix[route[i-1], route[i]] \
                                   + distance_matrix[route[i-1], c] \
                                   + distance_matrix[c, route[i]]
                        other_max = max(route_distances[:r_idx] + route_distances[r_idx+1:], default=0.0)
                        cand_max = max(new_dist, other_max)
                        if cand_max < best_max_val - 1e-12:
                            best_max_val = cand_max
                            best_customer = c
                            best_pos = (r_idx, i)
            if best_pos is None:
                break
            r_idx, i = best_pos
            routes[r_idx].insert(i, best_customer)
            route_distances[r_idx] = compute_route_distance(routes[r_idx])
            unassigned.remove(best_customer)
        return routes, route_distances
    
    def local_search(routes, route_distances):
        current_routes = [list(r) for r in routes]
        current_distances = list(route_distances)
        current_max = max(current_distances)
        max_iters = n * n
        for _ in range(max_iters):
            improved = False
            # Intra-2opt
            for r_idx in range(truck_count):
                route = current_routes[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = compute_route_distance(new_route)
                        if new_dist < current_distances[r_idx] - 1e-12:
                            current_distances[r_idx] = new_dist
                            current_routes[r_idx] = new_route
                            new_max = max(current_distances)
                            if new_max < current_max - 1e-12:
                                current_max = new_max
                                improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Inter-swap
            for r1 in range(truck_count):
                for r2 in range(r1+1, truck_count):
                    route1 = current_routes[r1]
                    route2 = current_routes[r2]
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new1 = route1[:i] + [route2[j]] + route1[i+1:]
                            new2 = route2[:j] + [route1[i]] + route2[j+1:]
                            new_dist1 = compute_route_distance(new1)
                            new_dist2 = compute_route_distance(new2)
                            other_max = max([current_distances[k] for k in range(truck_count) if k not in (r1, r2)] or [0])
                            cand_max = max(new_dist1, new_dist2, other_max)
                            if cand_max < current_max - 1e-12:
                                current_routes[r1] = new1
                                current_routes[r2] = new2
                                current_distances[r1] = new_dist1
                                current_distances[r2] = new_dist2
                                current_max = cand_max
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Inter-relocate
            for r1 in range(truck_count):
                for r2 in range(truck_count):
                    if r1 == r2:
                        continue
                    route1 = current_routes[r1]
                    route2 = current_routes[r2]
                    for i in range(1, len(route1)-1):
                        c = route1[i]
                        new1 = route1[:i] + route1[i+1:]
                        new_dist1 = compute_route_distance(new1)
                        best_new2 = None
                        best_new_dist2 = float('inf')
                        for j in range(1, len(route2)):
                            new2 = route2[:j] + [c] + route2[j:]
                            d2 = compute_route_distance(new2)
                            if d2 < best_new_dist2:
                                best_new_dist2 = d2
                                best_new2 = new2
                        other_max = max([current_distances[k] for k in range(truck_count) if k not in (r1, r2)] or [0])
                        cand_max = max(new_dist1, best_new_dist2, other_max)
                        if cand_max < current_max - 1e-12:
                            current_routes[r1] = new1
                            current_routes[r2] = best_new2
                            current_distances[r1] = new_dist1
                            current_distances[r2] = best_new_dist2
                            current_max = cand_max
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return current_routes, current_distances, current_max
    
    def ruin_recreate(routes, route_distances):
        new_routes = [list(r) for r in routes]
        new_distances = list(route_distances)
        # Find longest route
        max_idx = max(range(truck_count), key=lambda i: new_distances[i])
        route = new_routes[max_idx]
        if len(route) <= 2:
            return new_routes, new_distances
        # Remove 20% of interior customers randomly
        interior = list(range(1, len(route)-1))
        if len(interior) == 0:
            return new_routes, new_distances
        num_remove = max(1, int(0.2 * len(interior)))
        random.shuffle(interior)
        to_remove = interior[:num_remove]
        to_remove.sort(reverse=True)
        removed = []
        for pos in to_remove:
            removed.append(route.pop(pos))
        new_distances[max_idx] = compute_route_distance(route)
        # Reinsert with cheapest insertion minimizing max
        unassigned = set(removed)
        while unassigned:
            best_customer = None
            best_max_val = float('inf')
            best_pos = None
            for c in unassigned:
                for r_idx, r in enumerate(new_routes):
                    for i in range(1, len(r)):
                        new_dist = new_distances[r_idx] - distance_matrix[r[i-1], r[i]] \
                                   + distance_matrix[r[i-1], c] \
                                   + distance_matrix[c, r[i]]
                        other_max = max(new_distances[:r_idx] + new_distances[r_idx+1:], default=0.0)
                        cand_max = max(new_dist, other_max)
                        if cand_max < best_max_val - 1e-12:
                            best_max_val = cand_max
                            best_customer = c
                            best_pos = (r_idx, i)
            if best_pos is None:
                break
            r_idx, i = best_pos
            new_routes[r_idx].insert(i, best_customer)
            new_distances[r_idx] = compute_route_distance(new_routes[r_idx])
            unassigned.remove(best_customer)
        return new_routes, new_distances
    
    best_routes = None
    best_max_val = float('inf')
    max_restarts = 10
    for restart in range(max_restarts):
        if restart == 0:
            routes, route_distances = construction()
        else:
            routes, route_distances = ruin_recreate(best_routes, [compute_route_distance(r) for r in best_routes])
        routes, route_distances, current_max = local_search(routes, route_distances)
        if current_max < best_max_val - 1e-12:
            best_max_val = current_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
    return best_routes