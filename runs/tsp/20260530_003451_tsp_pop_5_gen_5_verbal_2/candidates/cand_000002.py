import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)
    # start with two random cities
    unvisited = list(range(n))
    np.random.shuffle(unvisited)
    tour = [unvisited[0], unvisited[1]]
    unvisited = unvisited[2:]
    for _ in range(n - 2):
        # find best city and insertion position
        best_cost = np.inf
        best_city = None
        best_pos = None
        for city in unvisited:
            for i in range(len(tour)):
                # cost increase when inserting city between tour[i] and tour[(i+1)%len(tour)]
                a = tour[i]
                b = tour[(i+1) % len(tour)]
                increase = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
                if increase < best_cost:
                    best_cost = increase
                    best_city = city
                    best_pos = i
        # insert at best position
        tour.insert(best_pos + 1, best_city)
        unvisited.remove(best_city)
    return np.array(tour, dtype=np.int32)