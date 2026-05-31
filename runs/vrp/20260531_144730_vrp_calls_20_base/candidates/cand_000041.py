import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 0:
        return []
    customer_count = n - 1
    if truck_count <= 0:
        return []
    if truck_count >= customer_count:
        routes = [[0, 0] for _ in range(truck_count)]
        for i in range(1, n):
            routes[i - 1] = [0, i, 0]
        return routes

    def route_dist(route):
        d = 0.0
        for i in range(len(route) - 1):
            d += distance_matrix[route[i], route[i + 1]]
        return d

    # Precompute distances for DP
    def precompute(perm):
        m = len(perm)
        start_to_depot = np.array([distance_matrix[0, c] for c in perm])
        end_to_depot = np.array([distance_matrix[c, 0] for c in perm])
        cum_inter = np.zeros(m + 1)
        for i in range(1, m):
            cum_inter[i] = cum_inter[i-1] + distance_matrix[perm[i-1], perm[i]]
        cum_inter[m] = cum_inter[m-1]
        return start_to_depot, end_to_depot, cum_inter

    def seg_dist(l, r, start_to_depot, end_to_depot, cum_inter):
        if l > r:
            return 0.0
        return start_to_depot[l] + (cum_inter[r] - cum_inter[l]) + end_to_depot[r]

    def split_perm(perm, start_to_depot, end_to_depot, cum_inter):
        m = len(perm)
        K = truck_count
        INF = float('inf')
        dp = [[INF] * (m + 1) for _ in range(K + 1)]
        choice = [[-1] * (m + 1) for _ in range(K + 1)]
        dp[0][0] = 0.0
        for t in range(1, K + 1):
            for i in range(t, m + 1):
                best_val = INF
                best_j = -1
                for j in range(t - 1, i):
                    cand = max(dp[t-1][j], seg_dist(j, i-1, start_to_depot, end_to_depot, cum_inter))
                    if cand < best_val - 1e-12:
                        best_val = cand
                        best_j = j
                    elif abs(cand - best_val) < 1e-12 and best_j > j:
                        best_j = j
                dp[t][i] = best_val
                choice[t][i] = best_j
        routes = []
        i = m
        for t in range(K, 0, -1):
            j = choice[t][i]
            l = j
            r = i - 1
            if l > r:
                routes.append([0, 0])
            else:
                route = [0] + perm[l:r+1] + [0]
                routes.append(route)
            i = j
        routes.reverse()
        return routes

    def intra_opt(route):
        if len(route) <= 3:
            return route
        best = route[:]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best) - 2):
                for j in range(i + 1, len(best) - 1):
                    if j - i == 1:
                        continue
                    new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    if route_dist(new_route) < route_dist(best) - 1e-12:
                        best = new_route
                        improved = True
                        break
                if improved:
                    break
        return best

    def random_initial_perm():
        # nearest neighbor with random tie-breaking
        unvisited = list(range(1, n))
        perm = []
        current = 0
        while unvisited:
            # compute distances, pick smallest, break ties randomly
            min_dist = min(distance_matrix[current, c] for c in unvisited)
            candidates = [c for c in unvisited if abs(distance_matrix[current, c] - min_dist) < 1e-12]
            next_node = random.choice(candidates)
            perm.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        # 2-opt improvement on tour with depot
        def two_opt_tour(tour):
            best = [0] + tour[:] + [0]
            improved = True
            while improved:
                improved = False
                for i in range(1, len(best) - 2):
                    for j in range(i + 1, len(best) - 1):
                        if j - i == 1:
                            continue
                        new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                        if route_dist(new_route) < route_dist(best) - 1e-12:
                            best = new_route
                            improved = True
                            break
                    if improved:
                        break
            return best[1:-1]
        perm = two_opt_tour(perm)
        return perm

    def compute_routes_from_perm(perm):
        start_to_depot, end_to_depot, cum_inter = precompute(perm)
        routes = split_perm(perm, start_to_depot, end_to_depot, cum_inter)
        for idx in range(truck_count):
            routes[idx] = intra_opt(routes[idx])
        return routes

    best_routes = None
    best_max = float('inf')
    # Number of restarts: at most 5, but bounded by instance size
    num_restarts = min(5, max(1, customer_count // 20))
    for restart in range(num_restarts):
        perm = random_initial_perm()
        routes = compute_routes_from_perm(perm)
        current_max = max(route_dist(r) for r in routes)
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
        # Simulated annealing on permutation
        m = len(perm)
        # Temperature schedule: linear cooling from T_start to T_end
        T_start = 1.0
        T_end = 0.01
        max_iter = customer_count * truck_count * 2
        temp = T_start
        for iteration in range(max_iter):
            # Try a random swap
            i, j = random.sample(range(m), 2)
            if i > j:
                i, j = j, i
            new_perm = perm[:]
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            new_routes = compute_routes_from_perm(new_perm)
            new_max = max(route_dist(r) for r in new_routes)
            delta = new_max - current_max
            if delta < 0 or random.random() < np.exp(-delta / temp):
                perm = new_perm
                current_max = new_max
                if current_max < best_max - 1e-12:
                    best_max = current_max
                    best_routes = [r[:] for r in new_routes]
                    report_best_vrp(best_routes)
            # Cool down
            temp = T_start - (T_start - T_end) * (iteration / max_iter)
            # Optional perturbation if no improvement in many iterations (not implemented to keep simple)
    # Final guarantee: best_routes should be set, but just in case:
    if best_routes is None:
        # Fallback: simple nearest neighbor and split
        unvisited = list(range(1, n))
        perm = []
        current = 0
        while unvisited:
            next_node = min(unvisited, key=lambda x: distance_matrix[current, x])
            perm.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        # 2-opt
        def two_opt_tour(tour):
            best = [0] + tour[:] + [0]
            improved = True
            while improved:
                improved = False
                for i in range(1, len(best) - 2):
                    for j in range(i + 1, len(best) - 1):
                        if j - i == 1:
                            continue
                        new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                        if route_dist(new_route) < route_dist(best) - 1e-12:
                            best = new_route
                            improved = True
                            break
                    if improved:
                        break
            return best[1:-1]
        perm = two_opt_tour(perm)
        routes = compute_routes_from_perm(perm)
        best_max = max(route_dist(r) for r in routes)
        best_routes = routes
    return best_routes