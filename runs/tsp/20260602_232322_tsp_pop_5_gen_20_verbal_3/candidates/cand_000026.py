import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def calc_dist(tour):
        d = 0.0
        for i in range(n):
            d += distance_matrix[tour[i], tour[(i+1)%n]]
        return d

    # Nearest-neighbor initial tour
    start = 0
    tour_nn = [start]
    unvisited = set(range(1, n))
    current = start
    while unvisited:
        next_city = min(unvisited, key=lambda c: distance_matrix[current, c])
        tour_nn.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    best_tour = np.array(tour_nn)
    best_dist = calc_dist(best_tour)
    report_best_tour(best_tour)

    # ACO parameters
    num_ants = min(10, n)
    max_iter = min(20, n*2)
    alpha = 1.0
    beta = 2.0
    evap = 0.5
    Q = 10.0

    tau = np.ones((n, n)) * 0.1
    eta = 1.0 / (distance_matrix + 1e-10)
    np.fill_diagonal(eta, 0)

    for _ in range(max_iter):
        all_tours = []
        for _ in range(num_ants):
            visited = [random.randint(0, n-1)]
            for _ in range(n-1):
                i = visited[-1]
                unvisited = [j for j in range(n) if j not in visited]
                probs = []
                for j in unvisited:
                    probs.append((tau[i,j] ** alpha) * (eta[i,j] ** beta))
                probs = np.array(probs)
                probs /= probs.sum()
                next_city = random.choices(unvisited, weights=probs, k=1)[0]
                visited.append(next_city)
            all_tours.append(visited)

        for ant_tour in all_tours:
            dist = calc_dist(np.array(ant_tour))
            if dist < best_dist - 1e-10:
                best_dist = dist
                best_tour = np.array(ant_tour)
                report_best_tour(best_tour)

        # Pheromone update
        tau *= (1 - evap)
        best_list = best_tour.tolist()
        for i in range(n):
            j = best_list[(i+1)%n]
            tau[i,j] += Q / best_dist
            tau[j,i] += Q / best_dist

    return best_tour