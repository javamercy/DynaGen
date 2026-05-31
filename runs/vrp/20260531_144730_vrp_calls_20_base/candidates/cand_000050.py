import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    random.seed(0)  # deterministic for reproducibility, can be removed for more randomness
    
    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def best_insertion(c, routes, route_dists):
        best = (float('inf'), -1, -1)
        second = (float('inf'), -1, -1)
        for r_idx, route in enumerate(routes):
            if len(route) < 2:
                continue
            other_max = 0.0
            for j, d in enumerate(route_dists):
                if j != r_idx and d > other_max:
                    other_max = d
            for pos in range(1, len(route)):
                pred = route[pos-1]
                succ = route[pos]
                new_dist = route_dists[r_idx] - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
                new_max = max(other_max, new_dist)
                if new_max < best[0]:
                    best, second = (new_max, r_idx, pos), best
                elif new_max < second[0]:
                    second = (new_max, r_idx, pos)
        return best[0], best[1], best[2], second[0]

    def construct_routes():
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        unassigned = list(range(1, n))
        while unassigned:
            candidates = []
            for c in unassigned:
                best_new_max, best_route, best_pos, second_new_max = best_insertion(c, routes, route_dists)
                if best_route == -1:
                    continue
                regret = second_new_max - best_new_max if second_new_max != float('inf') else float('inf')
                candidates.append((regret, c, best_route, best_pos, best_new_max))
            if not candidates:
                break
            # Sort by regret descending
            candidates.sort(key=lambda x: -x[0])
            # Randomly select among top 3 (or all if less)
            top_k = min(3, len(candidates))
            idx = random.randint(0, top_k-1)
            _, c, best_route, best_pos, new_max = candidates[idx]
            routes[best_route].insert(best_pos, c)
            route_dists[best_route] = route_dist(routes[best_route])
            unassigned.remove(c)
            report_best_vrp(routes)
        return routes, route_dists

    def intra_2opt(routes, route_dists, affected):
        for r_idx in affected:
            improved = True
            while improved:
                improved = False
                route = routes[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                        new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                        if new < old - 1e-12:
                            route[i:j+1] = reversed(route[i:j+1])
                            improved = True
                            route_dists[r_idx] = route_dist(route)
                            break
                    if improved:
                        break
        return routes, route_dists

    def inter_relocate(routes, route_dists):
        best_move = None
        best_new_max = max(route_dists)
        for from_idx in range(truck_count):
            from_route = routes[from_idx]
            for i in range(1, len(from_route)-1):
                c = from_route[i]
                pred = from_route[i-1]
                succ = from_route[i+1]
                new_from_dist = route_dists[from_idx] - distance_matrix[pred, c] - distance_matrix[c, succ] + distance_matrix[pred, succ]
                for to_idx in range(truck_count):
                    if to_idx == from_idx:
                        continue
                    to_route = routes[to_idx]
                    for pos in range(1, len(to_route)):
                        pred_o = to_route[pos-1]
                        succ_o = to_route[pos]
                        new_to_dist = route_dists[to_idx] - distance_matrix[pred_o, succ_o] + distance_matrix[pred_o, c] + distance_matrix[c, succ_o]
                        other_max = 0.0
                        for j, d in enumerate(route_dists):
                            if j != from_idx and j != to_idx and d > other_max:
                                other_max = d
                        new_overall = max(other_max, new_from_dist, new_to_dist)
                        if new_overall < best_new_max - 1e-12:
                            best_new_max = new_overall
                            best_move = (from_idx, i, to_idx, pos, new_from_dist, new_to_dist)
        return best_move, best_new_max

    # Initial construction
    best_routes = None
    best_max = float('inf')
    
    for restart in range(5):  # restart loop for exploration
        routes, route_dists = construct_routes()
        # Intensification: intra 2-opt on all routes
        routes, route_dists = intra_2opt(routes, route_dists, list(range(truck_count)))
        cur_max = max(route_dists)
        if cur_max < best_max - 1e-12:
            best_max = cur_max
            best_routes = [route[:] for route in routes]
        
        # Perturbation loop: random relocate moves followed by improvement
        for iter in range(20):  # bounded as instance size * truck_count
            # Random perturbation: perform a random relocate between two random routes
            from_idx = random.randint(0, truck_count-1)
            to_idx = random.randint(0, truck_count-1)
            while to_idx == from_idx:
                to_idx = random.randint(0, truck_count-1)
            from_route = routes[from_idx]
            if len(from_route) <= 3:  # must have at least one customer
                continue
            i = random.randint(1, len(from_route)-2)
            c = from_route[i]
            pred = from_route[i-1]
            succ = from_route[i+1]
            new_from_dist = route_dists[from_idx] - distance_matrix[pred, c] - distance_matrix[c, succ] + distance_matrix[pred, succ]
            to_route = routes[to_idx]
            pos = random.randint(1, len(to_route)-1)
            pred_o = to_route[pos-1]
            succ_o = to_route[pos]
            new_to_dist = route_dists[to_idx] - distance_matrix[pred_o, succ_o] + distance_matrix[pred_o, c] + distance_matrix[c, succ_o]
            # Apply move unconditionally (even if worsens) to diversify
            from_route.pop(i)
            to_route.insert(pos, c)
            route_dists[from_idx] = new_from_dist
            route_dists[to_idx] = new_to_dist
            # Intra 2-opt on affected routes
            routes, route_dists = intra_2opt(routes, route_dists, [from_idx, to_idx])
            cur_max = max(route_dists)
            if cur_max < best_max - 1e-12:
                best_max = cur_max
                best_routes = [route[:] for route in routes]
                report_best_vrp(routes)
            # Also try improving via inter-relocate after perturbation
            for _ in range(10):  # bounded improvement steps
                best_move, new_max = inter_relocate(routes, route_dists)
                if best_move is None:
                    break
                from_idx, i, to_idx, pos, new_from_dist, new_to_dist = best_move
                c = routes[from_idx].pop(i)
                routes[to_idx].insert(pos, c)
                route_dists[from_idx] = new_from_dist
                route_dists[to_idx] = new_to_dist
                routes, route_dists = intra_2opt(routes, route_dists, [from_idx, to_idx])
                cur_max = max(route_dists)
                if cur_max < best_max - 1e-12:
                    best_max = cur_max
                    best_routes = [route[:] for route in routes]
                    report_best_vrp(routes)
    
    return best_routes