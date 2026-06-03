import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    best_tour = None
    best_dist = float('inf')
    for _ in range(20):
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        current = start
        while unvisited:
            nearest = min(unvisited, key=lambda x: distance_matrix[current, x])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        tour = np.array(tour)
        # 2-opt improvement
        improved = True
        while improved:
            improved = False
            for i in range(n - 2):
                for j in range(i + 2, n):
                    if j - i == 1:
                        continue
                    a, b = tour[i], tour[i+1]
                    c, d = tour[j], tour[(j+1)%n]
                    old = distance_matrix[a, b] + distance_matrix[c, d]
                    new = distance_matrix[a, c] + distance_matrix[b, d]
                    if new < old:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
        dist = sum(distance_matrix[tour[k], tour[(k+1)%n]] for k in range(n))
        if dist < best_dist:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
    return best_tour