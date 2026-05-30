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

    # 2-opt first improvement
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

    tour = two_opt(tour)
    current_dist = np.sum(distance_matrix[tour[:-1], tour[1:]]) + distance_matrix[tour[-1], tour[0]]
    if current_dist < best_dist - 1e-12:
        best_dist = current_dist
        best_tour = tour.copy()
        report_best_tour(best_tour)

    # ILS with random 2-opt perturbation
    max_iter = 10
    for _ in range(max_iter):
        # random perturbation: reverse a random segment
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i > j:
            i, j = j, i
        if i == j:
            continue
        perturbed = tour.copy()
        if j - i > 1:
            perturbed[i:j] = np.flip(perturbed[i:j])
        new_tour = two_opt(perturbed)
        new_dist = np.sum(distance_matrix[new_tour[:-1], new_tour[1:]]) + distance_matrix[new_tour[-1], new_tour[0]]
        if new_dist < current_dist - 1e-12:
            tour = new_tour
            current_dist = new_dist
            if current_dist < best_dist - 1e-12:
                best_dist = current_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)

    # Final deterministic 2-opt
    tour = two_opt(tour)
    current_dist = np.sum(distance_matrix[tour[:-1], tour[1:]]) + distance_matrix[tour[-1], tour[0]]
    if current_dist < best_dist - 1e-12:
        best_dist = current_dist
        best_tour = tour.copy()
        report_best_tour(best_tour)

    return best_tour