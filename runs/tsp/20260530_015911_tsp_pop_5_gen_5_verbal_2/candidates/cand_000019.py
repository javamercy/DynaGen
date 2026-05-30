import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    # Farthest insertion construction
    tour = [0]
    unvisited = set(range(1, n))
    start = 0
    farthest = max(range(1, n), key=lambda x: distance_matrix[start, x])
    tour.append(farthest)
    unvisited.remove(farthest)
    while unvisited:
        best_dist = -1
        best_city = None
        for city in unvisited:
            min_dist = min(distance_matrix[city, t] for t in tour)
            if min_dist > best_dist:
                best_dist = min_dist
                best_city = city
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

    def two_opt(tour):
        current = tour.copy()
        improved = True
        while improved:
            improved = False
            for i in range(n - 2):
                for j in range(i + 2, n):
                    a, b = current[i], current[(i+1) % n]
                    c, d = current[j], current[(j+1) % n]
                    delta = (distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d])
                    if delta < -1e-12:
                        current[i+1:j+1] = np.flip(current[i+1:j+1])
                        improved = True
                        break
                if improved:
                    break
        return current

    def double_bridge(tour):
        n = len(tour)
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        k = np.random.randint(0, n)
        l = np.random.randint(0, n)
        indices = sorted([i, j, k, l])
        seg1 = tour[indices[0]:indices[1]]
        seg2 = tour[indices[1]:indices[2]]
        seg3 = tour[indices[2]:indices[3]]
        seg4 = np.concatenate([tour[indices[3]:], tour[:indices[0]]])
        new_tour = np.concatenate([seg1, seg4, seg3, seg2])[::-1]
        return new_tour

    # Deterministic first-improvement 2-opt
    tour = two_opt(tour)
    current_dist = np.sum(distance_matrix[tour[:-1], tour[1:]]) + distance_matrix[tour[-1], tour[0]]
    if current_dist < best_dist - 1e-12:
        best_dist = current_dist
        best_tour = tour.copy()
        report_best_tour(best_tour)

    # Iterated Local Search
    max_iter = 50
    for _ in range(max_iter):
        perturbed = double_bridge(tour)
        new_tour = two_opt(perturbed)
        new_dist = np.sum(distance_matrix[new_tour[:-1], new_tour[1:]]) + distance_matrix[new_tour[-1], new_tour[0]]
        if new_dist < current_dist - 1e-12:
            tour = new_tour
            current_dist = new_dist
            if current_dist < best_dist - 1e-12:
                best_dist = current_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)

    # Final deterministic 2-opt pass
    tour = two_opt(tour)
    current_dist = np.sum(distance_matrix[tour[:-1], tour[1:]]) + distance_matrix[tour[-1], tour[0]]
    if current_dist < best_dist - 1e-12:
        best_dist = current_dist
        best_tour = tour.copy()
        report_best_tour(best_tour)

    return best_tour