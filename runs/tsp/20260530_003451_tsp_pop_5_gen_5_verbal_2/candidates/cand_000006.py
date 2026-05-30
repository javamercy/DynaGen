import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        return np.arange(n, dtype=np.int32)
    # Cheapest insertion construction
    unvisited = list(range(n))
    np.random.shuffle(unvisited)
    tour = [unvisited[0], unvisited[1]]
    unvisited = unvisited[2:]
    for _ in range(n - 2):
        best_cost = np.inf
        best_city = None
        best_pos = None
        for city in unvisited:
            for i in range(len(tour)):
                a = tour[i]
                b = tour[(i + 1) % len(tour)]
                increase = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
                if increase < best_cost:
                    best_cost = increase
                    best_city = city
                    best_pos = i
        tour.insert(best_pos + 1, best_city)
        unvisited.remove(best_city)
    tour = np.array(tour, dtype=np.int32)
    best_tour = tour.copy()
    # Report initial tour
    report_best_tour(best_tour)
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        n_nodes = len(tour)
        for i in range(n_nodes):
            for j in range(i + 2, n_nodes):
                new_tour = tour.copy()
                new_tour[i+1:j+1] = tour[i+1:j+1][::-1]
                # Compute new distance (minimize distance_matrix calls)
                new_dist = 0.0
                for k in range(n_nodes):
                    new_dist += distance_matrix[new_tour[k], new_tour[(k+1)%n_nodes]]
                if new_dist < best_dist:
                    best_dist = new_dist
                    best_tour = new_tour.copy()
                    improved = True
                    report_best_tour(best_tour)
        tour = best_tour
    return best_tour