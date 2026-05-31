import numpy as np
from itertools import combinations
import random
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    n_cust = n - 1
    if truck_count >= n_cust:
        routes = [[0,0] for _ in range(truck_count)]
        for i in range(1, n):
            routes[i-1] = [0, i, 0]
        return routes

    # Farthest-first seeding with random tie-breaking
    def farthest_first_seeding():
        centers = [np.argmax(distance_matrix[0, 1:]) + 1]
        for _ in range(1, truck_count):
            dist_to_centers = np.min([[distance_matrix[c, i] for i in range(1, n)] for c in centers], axis=0)
            # find candidates with max distance, break ties randomly
            max_dist = np.max(dist_to_centers)
            candidates = np.where(dist_to_centers == max_dist)[0] + 1
            new_center = random.choice(candidates.tolist())
            centers.append(new_center)
        return np.array(centers)

    centers = farthest_first_seeding()

    # Assign each customer to nearest center (random tie)
    clusters = [[] for _ in range(truck_count)]
    for cust in range(1, n):
        dists = [distance_matrix[center, cust] for center in centers]
        min_dist = min(dists)
        candidates = [i for i, d in enumerate(dists) if d == min_dist]
        cluster_idx = random.choice(candidates)
        clusters[cluster_idx].append(cust)

    # Building initial routes: nearest neighbor with random selection among k=3 nearest
    def nearest_neighbor_route(nodes):
        if not nodes:
            return [0,0]
        route = [0]
        remaining = set(nodes)
        current = 0
        while remaining:
            # find k nearest (up to 3)
            sorted_candidates = sorted(remaining, key=lambda x: distance_matrix[current, x])
            k = min(3, len(sorted_candidates))
            next_node = random.choice(sorted_candidates[:k])
            route.append(next_node)
            remaining.remove(next_node)
            current = next_node
        route.append(0)
        return route

    def route_distance(route):
        total = 0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    def two_opt(route):
        best = route[:]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best)-2):
                for j in range(i+1, len(best)-1):
                    if j - i == 1:
                        continue
                    new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    if route_distance(new_route) < route_distance(best):
                        best = new_route
                        improved = True
                        break
                if improved:
                    break
        return best

    routes = []
    for cluster in clusters:
        if not cluster:
            routes.append([0,0])
        else:
            route = nearest_neighbor_route(cluster)
            route = two_opt(route)
            routes.append(route)

    best_routes = [r[:] for r in routes]
    report_best_vrp(best_routes)

    def max_route_distance(routes):
        return max(route_distance(r) for r in routes)

    best_max = max_route_distance(best_routes)

    # Simulated annealing parameters
    T0 = 100.0
    T_end = 1e-3
    cooling = 0.95
    max_outer = 10  # bounded outer loops
    max_inner = n_cust * 5  # bounded inner

    for outer in range(max_outer):
        T = T0
        while T > T_end:
            for _ in range(max_inner):
                # pick random move: relocate or exchange
                if random.random() < 0.5:
                    # relocate
                    from_route_idx = random.randrange(truck_count)
                    to_route_idx = random.randrange(truck_count)
                    if from_route_idx == to_route_idx:
                        continue
                    route_from = routes[from_route_idx]
                    route_to = routes[to_route_idx]
                    if len(route_from) <= 2:
                        continue
                    cust_pos = random.randint(1, len(route_from)-2)
                    insert_pos = random.randint(1, len(route_to)-1) if len(route_to)>1 else 1
                    cust = route_from[cust_pos]
                    new_routes = [r[:] for r in routes]
                    new_routes[from_route_idx].pop(cust_pos)
                    if len(new_routes[to_route_idx]) == 2 and new_routes[to_route_idx][0]==0 and new_routes[to_route_idx][1]==0:
                        new_routes[to_route_idx] = [0, cust, 0]
                    else:
                        new_routes[to_route_idx].insert(insert_pos, cust)
                else:
                    # exchange
                    r1, r2 = random.sample(range(truck_count), 2)
                    route1 = routes[r1]
                    route2 = routes[r2]
                    if len(route1) <= 2 or len(route2) <= 2:
                        continue
                    p1 = random.randint(1, len(route1)-2)
                    p2 = random.randint(1, len(route2)-2)
                    new_routes = [r[:] for r in routes]
                    cust1 = new_routes[r1][p1]
                    cust2 = new_routes[r2][p2]
                    new_routes[r1][p1] = cust2
                    new_routes[r2][p2] = cust1

                new_max = max_route_distance(new_routes)
                delta = new_max - best_max
                if delta < 0 or random.random() < math.exp(-delta/T):
                    # accept
                    routes = new_routes
                    # apply 2-opt to affected routes
                    routes[from_route_idx] = two_opt(routes[from_route_idx])
                    routes[to_route_idx] = two_opt(routes[to_route_idx])
                    if new_max < best_max:
                        best_routes = [r[:] for r in routes]
                        best_max = new_max
                        report_best_vrp(best_routes)
            T *= cooling

        # restart: shuffle cluster assignments
        if outer < max_outer-1:
            # reassign customers randomly to clusters (but maintain truck count)
            new_clusters = [[] for _ in range(truck_count)]
            for cust in range(1, n):
                cl = random.randrange(truck_count)
                new_clusters[cl].append(cust)
            new_routes = []
            for cluster in new_clusters:
                if not cluster:
                    new_routes.append([0,0])
                else:
                    route = nearest_neighbor_route(cluster)
                    route = two_opt(route)
                    new_routes.append(route)
            # compare with best
            new_max = max_route_distance(new_routes)
            if new_max < best_max:
                best_routes = [r[:] for r in new_routes]
                best_max = new_max
                report_best_vrp(best_routes)
            routes = new_routes  # even if worse, we diversify

    return best_routes