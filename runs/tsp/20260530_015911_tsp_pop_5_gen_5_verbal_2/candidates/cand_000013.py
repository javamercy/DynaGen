import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    # Farthest insertion construction
    tour = [0]
    unvisited = set(range(1, n))
    farthest = max(unvisited, key=lambda x: distance_matrix[0, x])
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

    def two_opt(current):
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

    # Initial 2-opt
    tour = two_opt(tour.copy())
    current_dist = np.sum(distance_matrix[tour[:-1], tour[1:]]) + distance_matrix[tour[-1], tour[0]]
    if current_dist < best_dist - 1e-12:
        best_dist = current_dist
        best_tour = tour.copy()
        report_best_tour(best_tour)

    # Simple double-bridge perturbation
    def double_bridge(tour):
        arr = tour.copy()
        n = len(arr)
        # Choose four random split points
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        k = np.random.randint(0, n)
        l = np.random.randint(0, n)
        indices = sorted([i, j, k, l])
        a, b, c, d = indices
        seg1 = arr[a:b]
        seg2 = arr[b:c]
        seg3 = arr[c:d]
        seg4 = np.concatenate([arr[d:], arr[:a]])
        # Reorder: seg1, seg4, seg3, seg2 (but also reverse one)
        new_tour = np.concatenate([seg1, seg4, seg3, seg2])
        return new_tour

    # Iterated Local Search with limited iterations
    for _ in range(10):
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
        # else keep current tour (no acceptance of worse)

    # Final deterministic 2-opt pass
    tour = two_opt(tour.copy())
    current_dist = np.sum(distance_matrix[tour[:-1], tour[1:]]) + distance_matrix[tour[-1], tour[0]]
    if current_dist < best_dist - 1e-12:
        best_dist = current_dist
        best_tour = tour.copy()
        report_best_tour(best_tour)

    return best_tour