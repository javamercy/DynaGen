import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0])
        report_best_tour(tour)
        return tour

    # nearest neighbor initial tour
    start = 0
    unvisited = set(range(1, n))
    tour = [start]
    cur = start
    while unvisited:
        next_city = min(unvisited, key=lambda x: distance_matrix[cur, x])
        tour.append(next_city)
        unvisited.remove(next_city)
        cur = next_city
    tour = np.array(tour, dtype=np.int32)
    best_tour = tour.copy()
    best_dist = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    report_best_tour(best_tour)

    # Simulated annealing parameters
    initial_temp = 0.2 * best_dist
    temp = initial_temp
    alpha = 0.99
    current_dist = best_dist
    current_tour = tour.copy()
    stagnant = 0
    max_stagnant = 500
    max_iterations = 10000
    for iteration in range(max_iterations):
        for _ in range(n * 10):
            i = np.random.randint(0, n-2)
            j = np.random.randint(i+2, n)
            a, b = current_tour[i], current_tour[(i+1)%n]
            c, d = current_tour[j], current_tour[(j+1)%n]
            delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
            if delta < 0 or np.random.random() < np.exp(-delta / temp):
                current_tour[i+1:j+1] = current_tour[i+1:j+1][::-1]
                current_dist += delta
                if current_dist < best_dist - 1e-12:
                    best_dist = current_dist
                    best_tour = current_tour.copy()
                    report_best_tour(best_tour)
                    stagnant = 0
        temp *= alpha
        stagnant += 1
        if stagnant >= max_stagnant:
            # restart from best
            current_tour = best_tour.copy()
            current_dist = best_dist
            temp = initial_temp
            stagnant = 0
        # early termination if temperature is very low
        if temp < 1e-6 * initial_temp:
            break
    return best_tour