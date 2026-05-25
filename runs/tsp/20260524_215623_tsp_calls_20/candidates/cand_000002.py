import numpy as np

def solve_tsp(distance_matrix, seed, budget):
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n, dtype=int)
        report_best_tour(tour)
        return tour
    rng = np.random.default_rng(seed)
    cities = list(range(n))
    rng.shuffle(cities)
    tour = [cities.pop(), cities.pop()]
    while cities:
        best_city = None
        best_pos = None
        best_cost = np.inf
        for city in cities:
            for i in range(len(tour)):
                a = tour[i]
                b = tour[(i+1)%len(tour)]
                delta = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
                if delta < best_cost - 1e-12:
                    best_cost = delta
                    best_city = city
                    best_pos = i + 1
                elif abs(delta - best_cost) < 1e-12:
                    if rng.random() < 0.5:
                        best_city = city
                        best_pos = i + 1
        tour.insert(best_pos, best_city)
        cities.remove(best_city)
    tour = np.array(tour, dtype=int)
    best_tour = tour.copy()
    best_len = tsp_length(distance_matrix, best_tour)
    report_best_tour(best_tour)
    if budget > 0:
        current_tour = tour.copy()
        current_len = best_len
        improvement = True
        while improvement and budget > 0:
            improvement = False
            for _ in range(min(20, budget)):
                if budget <= 0:
                    break
                i = rng.integers(n)
                j = rng.integers(n)
                if j == i or (j+1)%n == i or (i+1)%n == j:
                    continue
                if j < i:
                    i, j = j, i
                a = current_tour[i]
                b = current_tour[(i+1)%n]
                c = current_tour[j]
                d = current_tour[(j+1)%n]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                if delta < -1e-12:
                    current_tour[i+1:j+1] = current_tour[i+1:j+1][::-1]
                    current_len += delta
                    improvement = True
                    if current_len < best_len - 1e-12:
                        best_len = current_len
                        best_tour = current_tour.copy()
                        report_best_tour(best_tour)
                budget -= 1
    return best_tour

def tsp_length(dm, tour):
    return dm[tour, np.roll(tour, -1)].sum()