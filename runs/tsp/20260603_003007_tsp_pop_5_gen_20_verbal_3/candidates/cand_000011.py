import numpy as np

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0], dtype=int)
        report_best_tour(tour)
        return tour
    # nearest neighbor construction
    tour = [0]
    unvisited = set(range(1, n))
    current = 0
    while unvisited:
        next_city = min(unvisited, key=lambda x: (distance_matrix[current][x], x))
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    tour = np.array(tour, dtype=int)
    best_tour = tour.copy()
    def total_distance(t):
        total = 0.0
        for i in range(n):
            total += distance_matrix[t[i]][t[(i+1)%n]]
        return total
    best_cost = total_distance(tour)
    report_best_tour(best_tour)
    # simulated annealing
    current_tour = tour.copy()
    current_cost = best_cost
    T0 = 0.1 * best_cost
    cooling_rate = 0.999
    num_temps = 1000
    for t in range(num_temps):
        T = T0 * (cooling_rate ** t)
        if T < 1e-8:
            break
        for _ in range(n):
            i = np.random.randint(0, n-1)
            j = np.random.randint(i+2, n)
            if i == 0 and j == n-1:
                continue
            a = current_tour[i]
            b = current_tour[(i+1) % n]
            c = current_tour[j]
            d = current_tour[(j+1) % n]
            delta = distance_matrix[a][c] + distance_matrix[b][d] - distance_matrix[a][b] - distance_matrix[c][d]
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                new_tour = current_tour.copy()
                new_tour[i+1:j+1] = new_tour[i+1:j+1][::-1]
                current_tour = new_tour
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_tour = current_tour.copy()
                    report_best_tour(best_tour)
    return best_tour