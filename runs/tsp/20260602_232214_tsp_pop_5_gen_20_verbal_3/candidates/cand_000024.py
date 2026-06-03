import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    best_tour = None
    best_dist = float('inf')

    for start in range(n):
        # nearest neighbor construction
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        current = start
        while unvisited:
            next_city = min(unvisited, key=lambda c: distance_matrix[current, c])
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        tour = np.array(tour)
        dist = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
        if dist < best_dist - 1e-10:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)

        # 2-opt improvement
        improved = True
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    a = best_tour[i]
                    b = best_tour[i+1]
                    c = best_tour[j]
                    d = best_tour[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                    if delta < -1e-10:
                        new_tour = best_tour.copy()
                        new_tour[i+1:j+1] = best_tour[j:i:-1]
                        new_dist = best_dist + delta
                        if new_dist < best_dist - 1e-10:
                            best_dist = new_dist
                            best_tour = new_tour
                            improved = True
                            report_best_tour(best_tour)

    return best_tour