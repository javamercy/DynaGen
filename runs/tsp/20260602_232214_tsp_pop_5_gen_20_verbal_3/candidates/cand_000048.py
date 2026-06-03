import numpy as np
import math
import random

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 3:
        return np.arange(n)
    
    best_dist = float('inf')
    best_tour = None
    num_runs = 3
    max_iter_per_run = 5000
    
    for _ in range(num_runs):
        # === Regret construction with random tie-breaking ===
        tour = [0]
        first = np.argmin(distance_matrix[0][1:]) + 1
        tour.append(first)
        unvisited = set(range(n)) - set(tour)
        while unvisited:
            best_insert = {}
            second_best = {}
            for city in unvisited:
                best = float('inf')
                sec = float('inf')
                best_idx = None
                for pos in range(len(tour)):
                    i = tour[pos]
                    j = tour[(pos+1)%len(tour)]
                    cost = distance_matrix[i][city] + distance_matrix[city][j] - distance_matrix[i][j]
                    if cost < best:
                        sec = best
                        best = cost
                        best_idx = pos
                    elif cost < sec:
                        sec = cost
                best_insert[city] = (best_idx, best)
                second_best[city] = sec if sec != float('inf') else best
            regret = {c: second_best[c] - best_insert[c][1] for c in unvisited}
            max_regret = max(regret.values())
            candidates = [c for c in unvisited if regret[c] == max_regret]
            best_cost = min(best_insert[c][1] for c in candidates)
            best_candidates = [c for c in candidates if abs(best_insert[c][1] - best_cost) < 1e-12]
            chosen = random.choice(best_candidates)
            idx, _ = best_insert[chosen]
            tour.insert(idx+1, chosen)
            unvisited.remove(chosen)
        current_tour = np.array(tour)
        current_dist = sum(distance_matrix[current_tour[i]][current_tour[(i+1)%n]] for i in range(n))
        if current_dist < best_dist - 1e-12:
            best_dist = current_dist
            best_tour = current_tour.copy()
            report_best_tour(best_tour)
        
        # === Simulated annealing with 2-opt ===
        T = 1000.0
        alpha = 0.995
        for iteration in range(max_iter_per_run):
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j or (i+1)%n == j or (j+1)%n == i:
                continue
            if i > j:
                i, j = j, i
            a = current_tour[i]
            b = current_tour[(i+1)%n]
            c = current_tour[j]
            d = current_tour[(j+1)%n]
            delta = distance_matrix[a][c] + distance_matrix[b][d] - (distance_matrix[a][b] + distance_matrix[c][d])
            if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / T)):
                new_tour = np.concatenate([current_tour[:i+1], current_tour[i+1:j+1][::-1], current_tour[j+1:]])
                new_dist = current_dist + delta
                if new_dist < best_dist - 1e-12:
                    best_dist = new_dist
                    best_tour = new_tour.copy()
                    report_best_tour(best_tour)
                current_tour = new_tour
                current_dist = new_dist
            T *= alpha
            if T < 1e-8:
                break
    return best_tour