import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    # start with two closest cities
    start = 0
    # find nearest to start
    d = distance_matrix[start]
    d[start] = np.inf
    nearest = np.argmin(d)
    tour = [start, nearest]
    unvisited = set(range(n)) - {start, nearest}
    # cheap insertion
    while unvisited:
        best_insertion_cost = np.inf
        best_city = None
        best_pos = None
        for city in unvisited:
            # compute insertion cost for each position
            for pos in range(len(tour)):
                prev = tour[pos]
                next = tour[(pos+1) % len(tour)]
                cost = distance_matrix[prev, city] + distance_matrix[city, next] - distance_matrix[prev, next]
                if cost < best_insertion_cost:
                    best_insertion_cost = cost
                    best_city = city
                    best_pos = pos + 1  # insert after pos
        # insert best_city at best_pos
        tour.insert(best_pos, best_city)
        unvisited.remove(best_city)
    tour = np.array(tour)
    report_best_tour(tour)
    return tour