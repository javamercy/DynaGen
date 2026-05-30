import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    # Farthest insertion construction
    tour = [0]
    unvisited = set(range(1, n))
    # farthest from start
    start = 0
    farthest = max(range(1, n), key=lambda x: distance_matrix[start, x])
    tour.append(farthest)
    unvisited.remove(farthest)
    while unvisited:
        # find farthest unvisited city from current tour
        best_dist = -1
        best_city = None
        for city in unvisited:
            min_dist = min(distance_matrix[city, t] for t in tour)
            if min_dist > best_dist:
                best_dist = min_dist
                best_city = city
        # insert best_city in best position
        best_pos = None
        best_inc = np.inf
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            inc = distance_matrix[tour[i], best_city] + distance_matrix[best_city, tour[j]] - distance_matrix[tour[i], tour[j]]
            if inc < best_inc:
                best_inc = inc
                best_pos = j
        tour.insert(best_pos, best_city)
        unvisited.remove(best_city)
    tour = np.array(tour, dtype=np.int32)
    best_tour = tour.copy()
    best_dist = np.sum(distance_matrix[best_tour[:-1], best_tour[1:]]) + distance_matrix[best_tour[-1], best_tour[0]]
    report_best_tour(best_tour)
    
    # Deterministic first-improvement 2-opt
    improved = True
    while improved:
        improved = False
        for i in range(n - 2):
            for j in range(i + 2, n):
                a, b = tour[i], tour[(i+1) % n]
                c, d = tour[j], tour[(j+1) % n]
                delta = (distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d])
                if delta < -1e-12:
                    tour[i+1:j+1] = np.flip(tour[i+1:j+1])
                    current_dist = np.sum(distance_matrix[tour[:-1], tour[1:]]) + distance_matrix[tour[-1], tour[0]]
                    if current_dist < best_dist - 1e-12:
                        best_dist = current_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    improved = True
                    break
            if improved:
                break
    
    # Simulated annealing
    T_start = np.max(distance_matrix) * n
    T_end = 1e-8
    max_iter = 100000
    cooling = lambda k: T_start * (T_end / T_start) ** (k / max_iter)
    current_dist = np.sum(distance_matrix[tour[:-1], tour[1:]]) + distance_matrix[tour[-1], tour[0]]
    for iteration in range(max_iter):
        T = cooling(iteration)
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i > j:
            i, j = j, i
        if j - i < 2:
            continue
        i_next = (i + 1) % n
        j_next = (j + 1) % n
        delta = (distance_matrix[tour[i], tour[j]] + distance_matrix[tour[i_next], tour[j_next]] - distance_matrix[tour[i], tour[i_next]] - distance_matrix[tour[j], tour[j_next]])
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            tour[i_next:j+1] = np.flip(tour[i_next:j+1])
            current_dist += delta
            if current_dist < best_dist - 1e-12:
                best_dist = current_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
    
    # Final deterministic 2-opt pass
    improved = True
    while improved:
        improved = False
        for i in range(n - 2):
            for j in range(i + 2, n):
                a, b = tour[i], tour[(i+1) % n]
                c, d = tour[j], tour[(j+1) % n]
                delta = (distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d])
                if delta < -1e-12:
                    tour[i+1:j+1] = np.flip(tour[i+1:j+1])
                    current_dist = np.sum(distance_matrix[tour[:-1], tour[1:]]) + distance_matrix[tour[-1], tour[0]]
                    if current_dist < best_dist - 1e-12:
                        best_dist = current_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    improved = True
                    break
            if improved:
                break
    return best_tour