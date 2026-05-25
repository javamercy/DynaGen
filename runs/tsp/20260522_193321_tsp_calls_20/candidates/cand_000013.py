import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=int)
        try:
            report_best_tour(tour)
        except:
            pass
        return tour
    
    cand_size = 30 if n < 80 else 20
    candidates = []
    for i in range(n):
        dists = distance_matrix[i]
        order = np.argsort(dists)
        order = order[order != i][:cand_size]
        candidates.append(set(order))
    
    def tour_length(t):
        total = 0.0
        for i in range(n):
            total += distance_matrix[t[i]][t[(i+1)%n]]
        return total
    
    def get_pos(t):
        pos = np.empty(n, dtype=int)
        for idx, node in enumerate(t):
            pos[node] = idx
        return pos
    
    def construct_tour(rng):
        start = rng.randrange(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            best_city = None
            best_regret = -1.0
            best_pos = None
            best_cost = None
            for city in unvisited:
                min_cost = float('inf')
                second_min = float('inf')
                best_idx = -1
                for idx in range(len(tour)):
                    if idx == len(tour) - 1:
                        cost = distance_matrix[tour[idx]][city] + distance_matrix[city][tour[0]] - distance_matrix[tour[idx]][tour[0]]
                    else:
                        cost = distance_matrix[tour[idx]][city] + distance_matrix[city][tour[idx+1]] - distance_matrix[tour[idx]][tour[idx+1]]
                    if cost < min_cost:
                        second_min = min_cost
                        min_cost = cost
                        best_idx = idx
                    elif cost < second_min:
                        second_min = cost
                regret = second_min - min_cost
                if regret > best_regret:
                    best_regret = regret
                    best_city = city
                    best_pos = best_idx
                    best_cost = min_cost
            tour.insert(best_pos + 1, best_city)
            unvisited.remove(best_city)
        return np.array(tour, dtype=int)
    
    rng = random.Random(seed)
    tour = construct_tour(rng)
    best_tour = tour.copy()
    best_dist = tour_length(tour)
    try:
        report_best_tour(best_tour)
    except:
        pass
    evals = 0
    restart_counter = 0
    
    while evals < budget:
        improved = True
        while evals < budget and improved:
            improved = False
            pos = get_pos(tour)
            # 2-opt phase
            pairs = []
            for i in range(n):
                a = tour[i]
                for b in candidates[a]:
                    j = pos[b]
                    if j <= i:
                        continue
                    if (i+1)%n == j or (j+1)%n == i:
                        continue
                    pairs.append((i, j))
            rng.shuffle(pairs)
            for i, j in pairs:
                if evals >= budget:
                    break
                a = tour[i]
                b = tour[(i+1)%n]
                c = tour[j]
                d = tour[(j+1)%n]
                old = distance_matrix[a][b] + distance_matrix[c][d]
                new = distance_matrix[a][c] + distance_matrix[b][d]
                delta = new - old
                evals += 1
                if delta < -1e-12:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    current_dist = tour_length(tour)
                    if current_dist < best_dist - 1e-12:
                        best_dist = current_dist
                        best_tour = tour.copy()
                        try:
                            report_best_tour(best_tour)
                        except:
                            pass
                    improved = True
                    break
            if improved:
                continue
            # relocation phase
            reloc_pairs = []
            for u in range(n):
                i = pos[u]
                for v in candidates[u]:
                    if v == u:
                        continue
                    k = pos[v]
                    if i == k:
                        continue
                    if (i-1)%n == k or (i+1)%n == k:
                        continue
                    reloc_pairs.append((u, v))
            rng.shuffle(reloc_pairs)
            for u, v in reloc_pairs:
                if evals >= budget:
                    break
                i = pos[u]
                k = pos[v]
                if i == k:
                    continue
                prev_u = tour[(i-1)%n]
                next_u = tour[(i+1)%n]
                prev_v = tour[k]
                next_v = tour[(k+1)%n]
                old = distance_matrix[prev_u][u] + distance_matrix[u][next_u] + distance_matrix[prev_v][next_v]
                new = distance_matrix[prev_u][next_u] + distance_matrix[prev_v][u] + distance_matrix[u][next_v]
                delta = new - old
                evals += 1
                if delta < -1e-12:
                    tour_list = tour.tolist()
                    tour_list.remove(u)
                    idx_v = tour_list.index(v)
                    tour_list.insert(idx_v+1, u)
                    tour = np.array(tour_list, dtype=int)
                    current_dist = tour_length(tour)
                    if current_dist < best_dist - 1e-12:
                        best_dist = current_dist
                        best_tour = tour.copy()
                        try:
                            report_best_tour(best_tour)
                        except:
                            pass
                    improved = True
                    break
        if evals >= budget:
            break
        restart_counter += 1
        new_seed = seed + restart_counter
        rng = random.Random(new_seed)
        tour = construct_tour(rng)
        current_dist = tour_length(tour)
        if current_dist < best_dist - 1e-12:
            best_dist = current_dist
            best_tour = tour.copy()
            try:
                report_best_tour(best_tour)
            except:
                pass
    return best_tour