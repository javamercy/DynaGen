import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 1:
        tour = np.array([0])
        report_best_tour(tour.copy())
        return tour
    # farthest insertion initialization
    # start with the longest edge
    i, j = np.unravel_index(np.argmax(distance_matrix), (n, n))
    if i > j:
        i, j = j, i
    tour = [i, j]
    unvisited = set(range(n)) - {i, j}
    while unvisited:
        # find farthest city from current tour
        max_dist = -1
        farthest = None
        for city in unvisited:
            # distance to current tour: min distance to any tour node
            dist = min(distance_matrix[city, t] for t in tour)
            if dist > max_dist:
                max_dist = dist
                farthest = city
        # insert farthest at best position
        best_cost = np.inf
        best_pos = 0
        for pos in range(len(tour) + 1):
            # cost increase if inserted at pos
            if pos == 0:
                a = tour[-1]
                b = tour[0]
            elif pos == len(tour):
                a = tour[-1]
                b = tour[0]
            else:
                a = tour[pos-1]
                b = tour[pos]
            cost_increase = distance_matrix[a, farthest] + distance_matrix[farthest, b] - distance_matrix[a, b]
            if cost_increase < best_cost:
                best_cost = cost_increase
                best_pos = pos
        tour.insert(best_pos, farthest)
        unvisited.remove(farthest)
    tour = np.array(tour, dtype=int)
    report_best_tour(tour.copy())
    # first-improvement 2-opt
    improved = True
    while improved:
        improved = False
        for i in range(n - 2):
            for j in range(i + 2, n):
                a, b = tour[i], tour[(i+1) % n]
                c, d = tour[j], tour[(j+1) % n]
                delta = (distance_matrix[a, c] + distance_matrix[b, d]
                         - distance_matrix[a, b] - distance_matrix[c, d])
                if delta < -1e-12:
                    tour[i+1:j+1] = np.flip(tour[i+1:j+1])
                    improved = True
                    report_best_tour(tour.copy())
                    break
            if improved:
                break
    return tour