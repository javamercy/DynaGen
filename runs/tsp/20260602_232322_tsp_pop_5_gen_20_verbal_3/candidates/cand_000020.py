import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        return np.arange(n)
    # best nearest neighbor from all start nodes
    best_tour = None
    best_dist = float('inf')
    for start in range(n):
        unvisited = set(range(n))
        tour = [start]
        unvisited.remove(start)
        current = start
        while unvisited:
            next_node = min(unvisited, key=lambda u: distance_matrix[current][u])
            tour.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        dist = sum(distance_matrix[tour[i-1]][tour[i]] for i in range(n))
        if dist < best_dist:
            best_dist = dist
            best_tour = tour[:]
            report_best_tour(best_tour)
    tour = best_tour[:]
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for k in range(i+2, n):
                j = i
                if k - j < 2:
                    continue
                a, b = tour[j], tour[(j+1)%n]
                c, d = tour[k], tour[(k+1)%n]
                old = distance_matrix[a][b] + distance_matrix[c][d]
                new = distance_matrix[a][c] + distance_matrix[b][d]
                if new < old:
                    tour[j+1:k+1] = reversed(tour[j+1:k+1])
                    dist = sum(distance_matrix[tour[i-1]][tour[i]] for i in range(n))
                    if dist < best_dist:
                        best_dist = dist
                        best_tour = tour[:]
                        report_best_tour(best_tour)
                    improved = True
                    break
            if improved:
                break
    return np.array(best_tour)