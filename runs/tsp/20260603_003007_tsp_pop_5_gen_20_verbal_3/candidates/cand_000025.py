import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    # nearest neighbor
    unvisited = set(range(1, n))
    tour = [0]
    current = 0
    while unvisited:
        next_city = min(unvisited, key=lambda c: distance_matrix[current, c])
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    best_tour = tour.copy()
    best_cost = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    report_best_tour(np.array(best_tour))
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n-2):
            for j in range(i+2, n):
                A, B = tour[i], tour[(i+1)%n]
                C, D = tour[j], tour[(j+1)%n]
                delta = -distance_matrix[A,B] - distance_matrix[C,D] + distance_matrix[A,C] + distance_matrix[B,D]
                if delta < -1e-10:
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    best_cost += delta
                    best_tour = tour.copy()
                    improved = True
                    report_best_tour(np.array(best_tour))
                    break
            if improved:
                break
    return np.array(best_tour)