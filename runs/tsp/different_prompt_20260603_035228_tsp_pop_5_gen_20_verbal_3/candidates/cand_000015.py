import numpy as np

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    # Nearest neighbor construction
    tour = np.zeros(n, dtype=int)
    unvisited = set(range(1, n))
    current = 0
    for i in range(1, n):
        next_city = min(unvisited, key=lambda x: distance_matrix[current, x])
        tour[i] = next_city
        unvisited.remove(next_city)
        current = next_city
    current_dist = total_distance(tour, distance_matrix)
    best_tour = tour.copy()
    best_dist = current_dist
    report_best_tour(best_tour)
    
    max_iter = n * 1000
    for _ in range(max_iter):
        i, j = np.random.randint(1, n, 2)
        if i > j:
            i, j = j, i
        if i == j:
            continue
        prev_i = tour[i-1] if i > 0 else tour[-1]
        next_j = tour[(j+1) % n]
        old_edges = distance_matrix[prev_i, tour[i]] + distance_matrix[tour[j], next_j]
        new_edges = distance_matrix[prev_i, tour[j]] + distance_matrix[tour[i], next_j]
        delta = new_edges - old_edges
        if delta < 0:
            tour[i:j+1] = tour[i:j+1][::-1]
            current_dist += delta
            if current_dist < best_dist:
                best_dist = current_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
    return best_tour

def total_distance(tour, dist):
    return dist[tour, np.roll(tour, -1)].sum()