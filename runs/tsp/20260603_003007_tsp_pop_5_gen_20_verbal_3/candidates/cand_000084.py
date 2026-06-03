import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    best_dist = np.inf
    best_tour = np.arange(n)
    for _ in range(20):
        start = np.random.randint(n)
        unvisited = set(range(n))
        unvisited.remove(start)
        tour = [start]
        cur = start
        while unvisited:
            next_city = min(unvisited, key=lambda x: distance_matrix[cur, x])
            tour.append(next_city)
            unvisited.remove(next_city)
            cur = next_city
        tour = np.array(tour)
        improved = True
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    if j - i == 1:
                        continue
                    a, b = tour[i], tour[(i+1)%n]
                    c, d = tour[j], tour[(j+1)%n]
                    if distance_matrix[a,c] + distance_matrix[b,d] < distance_matrix[a,b] + distance_matrix[c,d]:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
        dist = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
        if dist < best_dist:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
    return best_tour