import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    # Nearest neighbor construction
    tour = [0]
    unvisited = set(range(1, n))
    current = 0
    while unvisited:
        next_city = min(unvisited, key=lambda x: (distance_matrix[current, x], x))
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    tour = np.array(tour, dtype=int)
    best_tour = tour.copy()
    best_cost = _total_distance(distance_matrix, best_tour)
    report_best_tour(best_tour)
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                new_tour = tour.copy()
                new_tour[i+1:j+1] = tour[j:i:-1]
                new_cost = _total_distance(distance_matrix, new_tour)
                if new_cost < best_cost:
                    best_tour = new_tour
                    best_cost = new_cost
                    tour = best_tour.copy()
                    improved = True
                    report_best_tour(best_tour)
                    break
            if improved:
                break
    return best_tour

def _total_distance(dm, tour):
    n = len(tour)
    total = 0.0
    for i in range(n - 1):
        total += dm[tour[i], tour[i+1]]
    return total