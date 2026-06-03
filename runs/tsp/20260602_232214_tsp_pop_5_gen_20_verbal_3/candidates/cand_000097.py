import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    # Nearest neighbor
    tour = [0]
    visited = {0}
    current = 0
    for _ in range(n-1):
        next_city = min((v for v in range(n) if v not in visited), key=lambda v: distance_matrix[current, v])
        tour.append(next_city)
        visited.add(next_city)
        current = next_city
    tour = np.array(tour)
    best_tour = tour.copy()
    best_cost = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    report_best_tour(tour)
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n-2):
            for j in range(i+2, n):
                a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
                delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                if delta < -1e-12:
                    tour = np.concatenate([tour[:i+1], tour[i+1:j+1][::-1], tour[j+1:]])
                    improved = True
        current_cost = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
        if current_cost < best_cost - 1e-12:
            best_cost = current_cost
            best_tour = tour.copy()
            report_best_tour(tour)
    return best_tour