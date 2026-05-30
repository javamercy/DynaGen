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

    def two_opt(t):
        current = t.copy()
        improved = True
        while improved:
            improved = False
            for i in range(n - 2):
                for j in range(i + 2, n):
                    a, b = current[i], current[(i+1) % n]
                    c, d = current[j], current[(j+1) % n]
                    delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                    if delta < -1e-12:
                        current[i+1:j+1] = np.flip(current[i+1:j+1])
                        improved = True
                        break
                if improved:
                    break
        return current

    # Apply 2-opt to initial tour
    tour = two_opt(tour)
    current_dist = np.sum(distance_matrix[tour[:-1], tour[1:]]) + distance_matrix[tour[-1], tour[0]]
    if current_dist < best_dist - 1e-12:
        best_dist = current_dist
        best_tour = tour.copy()
        report_best_tour(best_tour)

    # Random restarts
    np.random.seed(42)  # deterministic seed for reproducibility
    num_restarts = 5
    for _ in range(num_restarts):
        # Generate random permutation and apply 2-opt
        perm = np.random.permutation(n).astype(np.int32)
        new_tour = two_opt(perm)
        new_dist = np.sum(distance_matrix[new_tour[:-1], new_tour[1:]]) + distance_matrix[new_tour[-1], new_tour[0]]
        if new_dist < best_dist - 1e-12:
            best_dist = new_dist
            best_tour = new_tour.copy()
            report_best_tour(best_tour)

    # Final deterministic 2-opt pass on best tour
    best_tour = two_opt(best_tour)
    final_dist = np.sum(distance_matrix[best_tour[:-1], best_tour[1:]]) + distance_matrix[best_tour[-1], best_tour[0]]
    # No need to update if equal, but ensure best is returned
    return best_tour