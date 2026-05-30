import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def total_dist(t):
        return sum(distance_matrix[t[i], t[(i+1)%n]] for i in range(n))

    def farthest_insertion(start_node):
        tour = [start_node]
        in_tour = {start_node}
        # find farthest node from start
        farthest_node = max(range(n), key=lambda x: distance_matrix[start_node][x] if x != start_node else -1)
        tour.append(farthest_node)
        in_tour.add(farthest_node)
        while len(tour) < n:
            # find farthest node not in tour
            candidates = []
            max_min_dist = -1
            for node in range(n):
                if node in in_tour:
                    continue
                min_dist = min(distance_matrix[node][t] for t in tour)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    candidates = [node]
                elif min_dist == max_min_dist:
                    candidates.append(node)
            best_node = random.choice(candidates) if candidates else -1
            # find best insertion position
            best_pos = -1
            best_increase = float('inf')
            for i in range(len(tour)):
                a = tour[i]
                b = tour[(i+1)%len(tour)]
                increase = distance_matrix[a][best_node] + distance_matrix[best_node][b] - distance_matrix[a][b]
                if increase < best_increase:
                    best_increase = increase
                    best_pos = i+1
            tour.insert(best_pos, best_node)
            in_tour.add(best_node)
        return tour

    def two_opt_steepest(tour):
        improved = True
        while improved:
            improved = False
            best_delta = 0
            best_i = best_j = -1
            for i in range(n):
                for j in range(i+2, n):
                    if i == 0 and j == n-1:
                        continue
                    a, b = tour[i], tour[(i+1)%n]
                    c, d = tour[j], tour[(j+1)%n]
                    delta = distance_matrix[a][c] + distance_matrix[b][d] - (distance_matrix[a][b] + distance_matrix[c][d])
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            if best_delta < 0:
                i, j = best_i, best_j
                tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
                improved = True
        return tour

    best_tour = None
    best_cost = float('inf')
    num_restarts = 3
    for _ in range(num_restarts):
        start = random.randint(0, n-1)
        tour = farthest_insertion(start)
        tour = two_opt_steepest(tour)
        cost = total_dist(tour)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour.copy()
            report_best_tour(np.array(best_tour))
        # ILS: perturb and re-optimize
        for _ in range(2):  # two ILS iterations per restart
            # random 2-opt move (perturbation)
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            # ensure valid 2-opt move
            if i > j:
                i, j = j, i
            if j - i < 2 or (i == 0 and j == n-1):
                continue
            tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
            cost = total_dist(tour)
            # optional: report if better but we will re-optimize
            tour = two_opt_steepest(tour)
            cost = total_dist(tour)
            if cost < best_cost:
                best_cost = cost
                best_tour = tour.copy()
                report_best_tour(np.array(best_tour))
    return np.array(best_tour)