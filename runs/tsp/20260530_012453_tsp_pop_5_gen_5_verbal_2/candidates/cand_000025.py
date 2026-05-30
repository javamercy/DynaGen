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
        # farthest from start
        farthest = max(range(n), key=lambda x: distance_matrix[start_node][x] if x != start_node else -1)
        tour.append(farthest)
        in_tour.add(farthest)
        while len(tour) < n:
            # find farthest not in tour
            max_min = -1
            candidates = []
            for node in range(n):
                if node in in_tour:
                    continue
                min_dist = min(distance_matrix[node][t] for t in tour)
                if min_dist > max_min:
                    max_min = min_dist
                    candidates = [node]
                elif min_dist == max_min:
                    candidates.append(node)
            best_node = random.choice(candidates)
            # best insertion position
            best_increase = float('inf')
            best_pos = -1
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

    def steepest_2opt(tour):
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

    def double_bridge(tour):
        # split tour into 4 segments and reorder
        n = len(tour)
        if n < 8:
            return tour
        # random cut points
        p1 = random.randint(1, n//4)
        p2 = p1 + random.randint(1, n//4)
        p3 = p2 + random.randint(1, n//4)
        if p3 >= n:
            p3 = n - 1
        a = tour[:p1]
        b = tour[p1:p2]
        c = tour[p2:p3]
        d = tour[p3:]
        # reorder: a, c, b, d (or other variant)
        new_tour = a + c + b + d
        # ensure valid tour (no duplicates)
        if len(set(new_tour)) != n:
            return tour
        return new_tour

    best_tour = None
    best_cost = float('inf')
    num_restarts = 30
    ils_iterations = 5  # per restart
    for _ in range(num_restarts):
        start = random.randint(0, n-1)
        tour = farthest_insertion(start)
        tour = steepest_2opt(tour)
        cost = total_dist(tour)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour.copy()
            report_best_tour(np.array(best_tour))
        # ILS loop
        for _ in range(ils_iterations):
            perturbed = double_bridge(tour)
            if len(set(perturbed)) != n:
                continue
            tour = steepest_2opt(perturbed)
            cost = total_dist(tour)
            if cost < best_cost:
                best_cost = cost
                best_tour = tour.copy()
                report_best_tour(np.array(best_tour))
    return np.array(best_tour)