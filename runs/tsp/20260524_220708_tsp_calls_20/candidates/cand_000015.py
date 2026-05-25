import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)
    rng = np.random.RandomState(seed)
    tour = [rng.randint(n)]
    unvisited = list(set(range(n)) - {tour[0]})
    while unvisited:
        farthest = None
        max_min_dist = -1.0
        for city in unvisited:
            min_dist = np.min(distance_matrix[city, tour])
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                farthest = city
        best_pos = None
        best_increase = np.inf
        for i in range(len(tour)):
            prev = tour[i]
            nxt = tour[(i + 1) % len(tour)]
            inc = distance_matrix[prev, farthest] + distance_matrix[farthest, nxt] - distance_matrix[prev, nxt]
            if inc < best_increase:
                best_increase = inc
                best_pos = i + 1
        tour.insert(best_pos, farthest)
        unvisited.remove(farthest)
    tour = np.array(tour, dtype=int)
    best_tour = tour.copy()
    best_dist = 0.0
    for i in range(n):
        best_dist += distance_matrix[tour[i], tour[(i + 1) % n]]
    report_best_tour(best_tour)
    if budget > 0:
        improved = True
        ops = 0
        while improved and ops < budget:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    i_next = (i + 1) % n
                    j_next = (j + 1) % n
                    old = distance_matrix[tour[i], tour[i_next]] + distance_matrix[tour[j], tour[j_next]]
                    new = distance_matrix[tour[i], tour[j]] + distance_matrix[tour[i_next], tour[j_next]]
                    if new < old:
                        tour[i_next:j+1] = tour[i_next:j+1][::-1]
                        improved = True
                        ops += 1
                        new_dist = 0.0
                        for k in range(n):
                            new_dist += distance_matrix[tour[k], tour[(k + 1) % n]]
                        if new_dist < best_dist:
                            best_dist = new_dist
                            best_tour = tour.copy()
                            report_best_tour(best_tour)
                        break
                if improved:
                    break
    return best_tour