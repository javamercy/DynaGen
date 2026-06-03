import numpy as np

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0], dtype=np.int32)
        report_best_tour(tour)
        return tour
    best_tour = None
    best_dist = np.inf
    for start in range(n):
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        current = start
        while unvisited:
            next_city = min(unvisited, key=lambda x: distance_matrix[current, x])
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        tour = np.array(tour, dtype=np.int32)
        dist = 0
        for i in range(n):
            dist += distance_matrix[tour[i], tour[(i+1)%n]]
        if dist < best_dist:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
    tour = best_tour.astype(np.int32)
    ext = np.empty(n+1, dtype=np.int32)
    ext[:n] = tour
    ext[n] = tour[0]
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i+2, n):
                delta = (distance_matrix[ext[i], ext[j]] +
                         distance_matrix[ext[i+1], ext[j+1]] -
                         distance_matrix[ext[i], ext[i+1]] -
                         distance_matrix[ext[j], ext[j+1]])
                if delta < -1e-12:
                    ext[i+1:j+1] = ext[i+1:j+1][::-1]
                    tour = ext[:n].copy()
                    improved = True
                    report_best_tour(tour)
                    break
            if improved:
                break
    return tour