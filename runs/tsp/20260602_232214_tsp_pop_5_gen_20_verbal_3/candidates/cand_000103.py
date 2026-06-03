import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 3:
        return np.arange(n)
    # Cheapest insertion construction
    tour = [0]
    unvisited = set(range(1, n))
    while unvisited:
        best_cost = np.inf
        best_city = None
        best_pos = None
        for city in unvisited:
            for pos in range(len(tour)):
                a = tour[pos]
                b = tour[(pos+1)%len(tour)]
                cost = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
                if cost < best_cost:
                    best_cost = cost
                    best_city = city
                    best_pos = pos + 1
        tour.insert(best_pos, best_city)
        unvisited.remove(best_city)
    best_tour = np.array(tour)
    best_dist = sum(distance_matrix[best_tour[i], best_tour[(i+1)%n]] for i in range(n))
    report_best_tour(best_tour)
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i+2, n):
                a, b = best_tour[i], best_tour[(i+1)%n]
                c, d = best_tour[j], best_tour[(j+1)%n]
                if distance_matrix[a, b] + distance_matrix[c, d] > distance_matrix[a, c] + distance_matrix[b, d]:
                    new_tour = np.concatenate([best_tour[:i+1], best_tour[i+1:j+1][::-1], best_tour[j+1:]])
                    new_dist = sum(distance_matrix[new_tour[k], new_tour[(k+1)%n]] for k in range(n))
                    if new_dist < best_dist:
                        best_tour = new_tour
                        best_dist = new_dist
                        report_best_tour(best_tour)
                        improved = True
    return best_tour